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

1. **Push the gh-pages branch** (after merging this PR):
   ```bash
   git push origin gh-pages
   ```
   Or use the provided script:
   ```bash
   ./deploy-gh-pages.sh
   ```

2. **Configure GitHub Pages** in repository settings:
   - Go to Settings → Pages
   - Source: Select `gh-pages` branch
   - Click Save

3. **Access your site** at:
   https://codenova-spark.github.io/farm-to-computer/

### Manual Push Required

The gh-pages branch has been created locally with all necessary files. Due to environment constraints, it requires a manual push after this PR is merged:

```bash
git checkout main
git push origin gh-pages
```

Or use the deployment script:
```bash
./deploy-gh-pages.sh
```

Once pushed, GitHub Pages will automatically serve the index.html file.
