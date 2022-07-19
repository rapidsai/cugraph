# Building Documentation

...._coming soon_

## Steps to follow:

...._coming soon_


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
