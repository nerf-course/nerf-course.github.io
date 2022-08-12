## Compiling / publishing

Visit the page [https://airtable.com/account#](https://airtable.com/account#) and
`Generate API key`. You can then add this key to either `.bash_profile` or `.bashrc` as (the key below is a dummy, use your own):

```
export AIRTABLE_API_KEY="key60OcgX5W85SoGd"
```

To check everythign works, try to compile the blog:
```
python3 -m pip install pyairtable
python3 -m pip install install tqdm
python3 compile.py
```

Then push your commit (with the updated `index.md`) to the main branch for it to be auto-published to [https://nerf-course.github.io](https://nerf-course.github.io)

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/nerf-course/nerf-course.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Jekyll Support

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
