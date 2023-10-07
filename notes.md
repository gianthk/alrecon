### Run [solara](https://solara.dev/api/file_browser) app on localhost
Run file containing app:
```commandline
solara run file_browser.py --host localhost
```

Build multi-page app
```commandline
cd alrecon
pip install -e .
```

Run it with
```commandline
solara run alrecon.pages --host localhost
```
---
### ImageJ and Python
- https://imagej.net/scripting/python
- https://github.com/imagej/pyimagej
- https://www.nature.com/articles/s41592-022-01655-4

Downside is that pyimagej can only display java images. means that every time you have to convert your numpy_array.to_java()
- https://www.youtube.com/watch?v=chue-u3RpBM&t=1920s

