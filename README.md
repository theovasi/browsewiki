# Browsewiki
An information retrieval tool for Wikipedia, the free encyclopedia.

Most tools that are used today to retrieve information from text data, are query based search engines that apply a version of K-Nearest Neighbor search. These tools demand from the user to have a very well defined information need that can be condensed in a few words. This method of information retrieval works similarly to searching the index at the back of a book for a specific word of interest.  

Browsewiki approaches the task differently, aiming to enable users to acquire information on a general topic or get an overview of the thematical structure of a text collection similarly to how the contents of a book provide an overview of what it is about. To do so, browsewiki utilizes a browsing method called [Scatter/Gather](https://pdfs.semanticscholar.org/1134/3448f8a817fa391e3a7897a95f975ad287a.pdf) which is based on text document [clustering](https://en.wikipedia.org/wiki/Cluster_analysis). To speed up the rather slow clustering process, [topic modeling](https://drive.google.com/file/d/0B35kEsbpEjqoeEJoemZ3MDhWcjQ/view?usp=sharing) is used, in order to reduce the number of dimensions of the document representations.

## Installation

Step by step instructions on how to run the demo web application.

First, clone the repo and create a Python virtual environment :

`git clone https://github.com/theovasi/browsewiki`
`cd browsewiki`
`python3.6 -m venv venv`

activate the virtual environment :

`. ./venv/bin/activate`

then install the required libraries :

`pip install -r requirements.txt`

After that, run the setup script. This script will handle pre-processing the text data, creating the necessary vector representations, clustering and topic modeling. Currently Browsewiki supports the English and Greek versions of Wikipedia.

`python3 setup.py english`


Finally, run the web app :

`python3 app.py`

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/theovasi/browsewiki/blob/master/LICENSE) file for details


