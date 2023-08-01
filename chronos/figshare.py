import hashlib
import json
import os
import requests
import zipfile

from requests.exceptions import HTTPError

# CONSTANTS
BASE_URL = 'https://api.figshare.com/v2/{endpoint}'
CHUNK_SIZE = 10485760

# FIGSHARE
FIGSHARE_TOKEN = 'c727b3826353164e0ae4ba35c9325ee339597f2ca66c3ab4349394cc0bcf2662b91f0a3e722f41b0d26152c65ed63692c834a18be6d7df8141d16e3fed834aa4'
FIGSHARE_ID = 21411663
MODEL_NAME = 'mini_model.zip'
# ^ credentials for smaffa and test article id


# FIGSHARE_TOKEN = <achilles-master-token>
# FIGSHARE_ID = <chronos-model-article-id> (14067047?)
# MODEL_NAME = <model-zip-file>


##### GENERAL API UTILS #####

def raw_issue_request(method, url, data=None, binary=False):
    '''
    Helper for issuing an HTTPS request
    '''
    headers = {'Authorization': 'token ' + FIGSHARE_TOKEN}
    if data is not None and not binary:
        data = json.dumps(data)
    response = requests.request(method, url, headers=headers, data=data)
    try:
        response.raise_for_status()
        try:
            data = json.loads(response.content)
        except ValueError:
            data = response.content
    except HTTPError as error:
        print('Caught an HTTPError: {}'.format(error.message))
        print('Body:\n', response.content)
        raise

    return data


def issue_request(method, endpoint, *args, **kwargs):
    '''
    Formats an HTTPS request
    '''
    return raw_issue_request(method, BASE_URL.format(endpoint=endpoint), *args, **kwargs)


##### UPLOADING #####

def zip_chronos_model(path, archive_name=None):
    '''
    Zip the necessary files for storing a Chronos model
    '''
    files_in_path = os.listdir(path)
    necessary_files = ['chronos_ge_unscaled.hdf5', 
                     'guide_efficacy.csv', 
                     'cell_line_efficacy.csv', 
                     'screen_delay.csv', 
                     'library_effect.csv']
    for filename in necessary_files:
        assert filename in files_in_path, "Cannot locate file {} in target directory {}".format(filename, path)
    
    if archive_name is None:
        archive_name = path.rstrip('/')
    
    with zipfile.ZipFile(archive_name + '.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as ziph:
        for filename in necessary_files:
            ziph.write(os.path.join(path, filename), 
                       os.path.relpath(os.path.join(archive_name, filename),
                                       os.path.join(path, '..')))
    return archive_name + '.zip'


def list_files_of_article(article_id, private=True):
    '''
    List all the files present in a figshare article
    '''
    if private:
        result = issue_request('GET', 'account/articles/{}/files'.format(article_id))
    else:
        result = issue_request('GET', 'articles/{}/files'.format(article_id))
    print('Listing files for article {}:'.format(article_id))
    if result:
        for item in result:
            print('  {id} - {name}'.format(**item))
    else:
        print('  No files.')


def create_article(title):
    '''
    Make a new figshare article
    '''
    data = {
        'title': title
    }
    result = issue_request('POST', 'account/articles', data=data)
    print('Created article:', result['location'], '\n')

    result = raw_issue_request('GET', result['location'])

    return result['id']


def get_file_check_data(file_name):
    '''
    Ensure file can be streamed for upload
    '''
    with open(file_name, 'rb') as fin:
        md5 = hashlib.md5()
        size = 0
        data = fin.read(CHUNK_SIZE)
        while data:
            size += len(data)
            md5.update(data)
            data = fin.read(CHUNK_SIZE)
        return md5.hexdigest(), size


def initiate_new_upload(article_id, file_name):
    '''
    Initiate the upload process for a file
    '''
    endpoint = 'account/articles/{}/files'
    endpoint = endpoint.format(article_id)

    md5, size = get_file_check_data(file_name)
    data = {'name': os.path.basename(file_name),
            'md5': md5,
            'size': size}

    result = issue_request('POST', endpoint, data=data)
    print('Initiated file upload:', result['location'], '\n')

    result = raw_issue_request('GET', result['location'])

    return result


def complete_upload(article_id, file_id):
    '''
    Complete the file upload
    '''
    issue_request('POST', 'account/articles/{}/files/{}'.format(article_id, file_id))


def upload_parts(file_info, file_name):
    '''
    Uploads an entire file in chunks
    '''
    url = '{upload_url}'.format(**file_info)
    result = raw_issue_request('GET', url)

    print('Uploading parts:')
    with open(file_name, 'rb') as fin:
        for part in result['parts']:
            upload_part(file_info, fin, part)


def upload_part(file_info, stream, part):
    '''
    Uploads a single chunk of a file
    '''
    udata = file_info.copy()
    udata.update(part)
    url = '{upload_url}/{partNo}'.format(**udata)

    stream.seek(part['startOffset'])
    data = stream.read(part['endOffset'] - part['startOffset'] + 1)

    raw_issue_request('PUT', url, data=data, binary=True)
    print('  Uploaded part {partNo} from {startOffset} to {endOffset}'.format(**part))


def upload(file_path, article_id=None, article_title=None, overwrite=False):
    '''
    Uploads a local file to the specified article, or creates a new article with the file
    '''
    # create article if not exists
    if article_id is None:
        assert article_title is not None, 'No article_id supplied, please provide a title for the new dataset or specify the id of an existing one'
        article_id = create_article(article_title) 
    else:
        # check if file exists
        response = issue_request('GET', 'account/articles/{article_id}'.format(article_id=article_id))
        file_list = response['files']
        
        for file_info in file_list:
            if file_info['name'] == os.path.basename(file_path):
                if overwrite:
                    # Delete the existing file first
                    issue_request('DELETE', 'account/articles/{article_id}/files/{file_id}'.format(article_id=article_id, file_id=file_info['id']))
                else:
                    # Throw an error
                    raise ValueError('{} exists in figshare article'.format(os.path.basename(file_path)))
                    

    # Upload the file
    file_info = initiate_new_upload(article_id, file_path)
    upload_parts(file_info, file_path)
    complete_upload(article_id, file_info['id'])
    
    list_files_of_article(article_id)


##### DOWNLOADING #####

def unzip(archive_path, target_path=None):
    '''
    Unzips a zip archive into the target directory
    '''
    if target_path is None:
        target_path = os.path.dirname(archive_path)
    
    with zipfile.ZipFile(archive_path, 'r') as ziph:
        ziph.extractall(target_path)
    
    return os.path.join(target_path, os.path.basename(archive_path).rstrip('.zip'))


def download_files_from_article(article_id, target_directory=None, fileset=None, private=False):
    '''
    Downloads files from a public (or private) Figshare article
    Parameters:
        article_id (`str` or `int`): identifier for a Figshare dataset
        target_directory (`str`): the location to download files into; if None, creates a local directory named by article_id
        fileset (iterable): Figshare file ids or names to download
    '''
        
    if private: # for test purposes
        response = issue_request('GET', 'account/articles/{article_id}'.format(article_id=article_id))
    else:
        response = issue_request('GET', 'articles/{article_id}'.format(article_id=article_id))
    
    headers = {'Authorization': 'token ' + FIGSHARE_TOKEN}
    
    file_list = response['files']
    
    if target_directory is None: # save the downloads by the article id
        target_directory = 'figshare_{}'.format(article_id)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for file_info in file_list:
        if file_info['id'] in fileset or file_info['name'] in fileset:
            r = requests.get('https://ndownloader.figshare.com/files/{file_id}'.format(file_id=file_info['id']), 
                             allow_redirects=True, headers=headers)
            with open(os.path.join(target_directory, file_info['name']), 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                print('Downloaded {} from article {}'.format(file_info['name'], article_id))
    print('Downloads are located at {}/'.format(target_directory))
    

# def download_files_from_article(article_id, target_directory=None, fileset=None):
#     '''
#     Downloads files from a public Figshare article
#     Parameters:
#         article_id (`str` or `int`): identifier for a Figshare dataset
#         target_directory (`str`): the location to download files into; if None, creates a local directory named by article_id
#         fileset (iterable): Figshare file ids or names to download
#     '''
        
#     response = issue_request('GET', 'articles/{article_id}'.format(article_id=article_id))
    
#     headers = {'Authorization': 'token ' + FIGSHARE_TOKEN}
    
#     file_list = response['files']
    
#     if target_directory is None: # save the downloads by the article id
#         target_directory = 'figshare_{}'.format(article_id)
#     if not os.path.exists(target_directory):
#         os.makedirs(target_directory)
    
#     for file_info in file_list:
#         if file_info['id'] in fileset or file_info['name'] in fileset:
#             r = requests.get('https://ndownloader.figshare.com/files/{file_id}'.format(file_id=file_info['id']), 
#                              allow_redirects=True, headers=headers)
#             with open(os.path.join(target_directory, file_info['name']), 'wb') as f:
#                 for chunk in r.iter_content(1024):
#                     f.write(chunk)
#                 print('Downloaded {} from article {}'.format(file_info['name'], article_id))
#     print('Downloads are located at {}'.format(target_directory))
                    