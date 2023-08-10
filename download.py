# -*- coding: utf-8 -*-
"""
Updated 4/5/21

@author: Cameron Cummins

Data download, concatenate, and formatting functions for Dr. Persad's Hydroclimate Research.
"""
import urllib.request
import requests
import os


def getFilesLinkedInHTML(head_url:str, sub_path:str="", filter_file:list=[".nc"]) -> list:
    """
    Indexes links contained in an HTML file specified by a URL and recursively explores 
    subdirectories also linked in the HTML file.
    
    Parameters
    ----------
    head_url : string
        Head URL to begin search at
    sub_path : string
        sub path to append to the links
    filter_file : list
        list of filters to apply when downloading the files linked in the HTML
        
    Returns
    -------
    file_paths : list of strings
        list of file paths
    """
    
    # Empty cache before indexing
    urllib.request.urlcleanup()
    # Open head directory
    html_data = open(str(urllib.request.urlretrieve(head_url + sub_path)[0]), 'r')
    # Create list for storing the paths to the files found
    file_paths = []
    
    # Iterate through the webpage
    for line in html_data:
        # Look for links
        if "<a href=" in line and not "../" in line:
            # Parse out the directory path from the link
            first_parse, second_parse = line.split('="', 1)
            directory, third_parse = second_parse.split('">', 1)
            # If the link points to a new directory, recursively explore this new directory
            if "/" in directory:
                file_paths += getFilesLinkedInHTML(head_url, sub_path + directory, filter_file)
            # If the link points to a file with a filter in its name, add it to the list of paths
            else:
                for a_filter in filter_file:
                    if a_filter in directory:
                        file_paths.append(str(sub_path + directory))
                        break
    # Close webpage and return paths found
    html_data.close()
    return file_paths

def outputFileIndex(file_paths:list, head_url:str="", output_dir:str="", output_file_name:str="file_index.in", mode:str='w') -> None:
    """
    Outputs a list of indexed paths to a file

    Parameters
    ----------
    file_paths : list
        url pointing to HTML file to index
    head_url : str, optional
        it may be convenient to add a path to the front of each path in the list. 
        For example, if you indexed a website but didn't preserve the domain, you may 
        wish to add "http://www.website.com/" to the front of each path. This will not 
        add it to the front of each URL, but will indicate at the top of the file the "head URL"
    output_dir : str, optional
        path to directory to output index to
    output_file_name : str, optional
        name of file to output index to
    mode : str, optional
        'w' for overriding existing file, 'a' for appending the existing file

    Returns
    -------
    None
    """
    with open(output_dir + output_file_name, mode) as file:
        file.write(head_url + "\n")
        for path in file_paths:
            file.write(path + "\n")

def readFileIndex(path_to_index:str) -> (str, list):
    """
    Create list from file containing indexed paths
    
    Parameters
    ----------
    path_to_index : str
        path to file containing index of paths

    Returns
    -------
    (str, list)
        tuple of the head url and a list of the indexed paths

    """
    file_paths = []
    head_url = ""
    with open(path_to_index, 'r') as file:
        head_url = file.readline()[::-1][1::][::-1]
        while True:
            line = file.readline()
            if line == '':
                break
            else:
                file_paths.append(line[::-1][1::][::-1])
    return (head_url, file_paths)

def downloadFromIndex(file_urls:list, output_dir:str="", preserve_structure:bool=True, head_directory:str="") -> (int, int):
    """
    Downloads files from list of URLs
    
    Parameters
    ----------
    file_urls : list
        url pointing to HTML file to index
    output_dir : str, optional
        path to directory to output downloaded files to
    preserve_structure : bool, optional
        whether or not to create the file structure described by the paths or to just download files into a single directory
    head_directory : str, optional
        if preserving the structure, you can choose which directory in the paths in the list to use as the top directory 
        (instead of, for example, using /website/ as the top directory)

    Returns
    -------
    (int, int)
        tuple of the number of files downloaded and the number of bytes downloaded

    """
    # Track number of files downloaded and total data size
    num_files = 0
    bytes_downloaded = 0
    
    # Clear cache
    urllib.request.urlcleanup()
    for file_url in file_urls:
        print("Downloading: " + file_url)
        # Download item to cache 
        download_item = requests.get(file_url)
        # Log size
        bytes_downloaded += int(download_item.headers['Content-Length'])
        # Generate path for outputting file to
        if preserve_structure:
            dir_structure = file_url[7::][::-1].split("/", 1)[1][::-1]
            if head_directory != "":
                dir_structure = dir_structure.split(head_directory)[1]
        else:
            # If not preserving structure, just output the file to this directory
            dir_structure = ""
        
        file_name = file_url[7::][::-1].split("/", 1)[0][::-1]
        
        output_path = output_dir + dir_structure
        
        output_path_to_file = output_path + "/" + file_name
        
        skip = False
        # Create directories if necessary
        if not (os.path.isdir(output_path)):
            os.makedirs(output_path)
            print("Making directory: " + output_path)
        elif (os.path.isfile(output_path_to_file)):
            if not os.path.getsize(output_path_to_file) == 0:
                print("Skipping: " + output_path_to_file)
                skip = True
        # Get file contents and write to disk
        if not skip:
            print("Writing to: " + output_path_to_file)
            file = open(output_path_to_file, 'wb')
            file.write(download_item.content)
            file.close()
        num_files += 1
    return num_files, bytes_downloaded

HEAD_URL = "http://albers.cnr.berkeley.edu/data/scripps/loca/met/"
full_paths = getFilesLinkedInHTML(HEAD_URL, filter_file=[".nc"])
models = ['ACCESS1-0', 'CCSM4', 'CESM1-BGC','CMCC-CMS','CNRM-CM5', 'CanESM2', 'GFDL-CM3','HadGEM2-CC','HadGEM2-ES','MIROC5']
paths = [HEAD_URL + path for path in full_paths if "historical" not in path and path.split('/')[0] in models and "pr" in path]
downloadFromIndex(paths,  head_directory="met/")
