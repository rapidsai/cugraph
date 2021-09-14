# Building Documentation

All prerequisite for building docs are in the cugraph development conda environment.
[See build instructions](../../SOURCEBUILD.md) on how to create the development conda environment

## Steps to follow:

In order to build the docs, we need the conda dev environment from cugraph and we need to build cugraph from source.  

1. Create a conda env and build cugraph from source. The dependencies to build rapids from source are installed in that conda environment, and then rapids is built and installed into the same environment.

2. Once cugraph is built from source, navigate to `../docs/cugraph/`. If you have your documentation written and want to turn it into HTML, run makefile:


```bash
# most be in the /docs/cugraph directory
make html
```

This should run Sphinx in your shell, and outputs to `build/html/index.html`


## View docs web page by opening HTML in browser:

First navigate to `/build/html/` folder, and then run the following command:

```bash
python -m http.server
```
Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

```
https://<host IP-Address>:8000
```
Now you can check if your docs edits formatted correctly, and read well.
