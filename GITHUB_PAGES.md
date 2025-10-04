# GitHub Pages Setup

## GH-Pages Branch

A `gh-pages` branch has been created locally with a simple HTML index page. This branch contains:

- `index.html` - A simple, styled web page showcasing the Farm to Computer project

### Branch Contents

The gh-pages branch includes:
- A responsive HTML page with project information
- Clean CSS styling for a professional appearance
- Project features and descriptions
- Links back to the main repository

### To Deploy

To enable GitHub Pages for this repository:
1. The gh-pages branch needs to be pushed to the remote repository
2. In the repository settings, configure GitHub Pages to use the gh-pages branch
3. The site will be available at: https://codenova-spark.github.io/farm-to-computer/

### Manual Push Required

Due to authentication constraints in this environment, the gh-pages branch was created locally but requires manual push:

```bash
git push origin gh-pages
```

Once pushed, GitHub Pages will automatically serve the index.html file.
