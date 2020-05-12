# Building Documentation

A basic python environment with packages listed in `./requirement.txt` is
enough to build the docs.

## Get additional dependency

```bash
pip install -r requirement.txt
```

## Run makefile:

```bash
make html
```

Outputs to `build/html/index.html`

## View docs web page by opening HTML in browser:

First navigate to `/build/html/` folder, i.e., `cd build/html` and then run the following command:

```bash
python -m http.server
```
Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

```
https://<host IP-Address>:8000
```
Now you can check if your docs edits formatted correctly, and read well.
