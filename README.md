# browsewiki
Browsewiki is a web app for browsing Wikipidea articles. The app utilizes the 
[Scatter/Gather](https://pdfs.semanticscholar.org/1134/3448f8a817fa391e3a7897a95f975ad2873a.pdf) search method. This 
method can be applied on any collection of text documents and consists of two steps. The first step is the Scatter step, 
where the documents are grouped into semantically similar groups (clusters), in a process called 
[clustering](https://en.wikipedia.org/wiki/Cluster_analysis). The second step is the Gather step where the user selects 
the groups of documents that interest him/her. In the next Scatter step only the documents that belong in the selected 
clusters will be clustered. These two steps are repeated until a single document of interest is reached.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/theovasi/browsewiki/blob/master/LICENSE) file for details


