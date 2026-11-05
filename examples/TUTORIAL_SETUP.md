# Running the eduACDC tutorial

The tutorial notebook [`tutorial.ipynb`](tutorial.ipynb) introduces eduACDC step by step, assuming no previous experience with Python or Jupyter. There are two ways to run it. Option A needs only a web browser and a Google account and is the easiest. Option B runs everything on your own computer.

## Option A: Google Colab (nothing to install)

1. Open the notebook in Colab: https://colab.research.google.com/github/edahelsinki/eduACDC/blob/main/examples/tutorial.ipynb
2. Sign in with a Google account if asked.
3. Choose *Runtime → Run all* from the menu, or run the cells one by one with **Shift + Enter**. Colab may warn that the notebook was not authored by Google; click *Run anyway*.
4. The first code cell installs eduACDC, which takes about a minute. After that each cell runs in seconds.

Changes you make are lost when you close the tab unless you save a copy with *File → Save a copy in Drive*.

## Option B: your own computer

You need to do steps 1 and 2 only once.

1. **Install uv**, a tool that installs Python and the required packages for you. Open a terminal (macOS: the *Terminal* app; Windows: *PowerShell*) and run

   macOS or Linux:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Windows:
   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Then close the terminal and open a new one. See https://docs.astral.sh/uv/ if something goes wrong.

2. **Download eduACDC.** Go to https://github.com/edahelsinki/eduACDC, click the green *Code* button, choose *Download ZIP*, and unzip the file. (If you use git: `git clone https://github.com/edahelsinki/eduACDC.git`.)

3. **Start the notebook.** In the terminal, go to the unzipped folder and start JupyterLab:
   ```
   cd path/to/eduACDC
   uv run --with jupyter jupyter lab examples/tutorial.ipynb
   ```
   Replace `path/to/eduACDC` with the actual location, for example `cd Downloads/eduACDC-main`. The first time, uv downloads Python and all packages, which takes a few minutes. JupyterLab then opens in your web browser.

4. Run the cells with **Shift + Enter**, or choose *Run → Run All Cells*. When you are done, close the browser tab and press **Ctrl + C** in the terminal to stop JupyterLab.

Next time, only step 3 is needed.

If you already work with Python, you can instead install eduACDC into a virtual environment with `pip install -e .` in the eduACDC folder, install JupyterLab with `pip install jupyterlab`, and open the notebook as usual.

## If something goes wrong

- **`NameError: name '...' is not defined`**: a cell above was not run. Run the notebook from the top, e.g. with *Run all*.
- **`ModuleNotFoundError: No module named 'eduacdc'`**: in Colab, the first code cell was not run or the installation failed; run it again. On your own computer, start JupyterLab with the `uv run` command above from the eduACDC folder, not in some other way.
- **Error when downloading `ANW.yaml` or installing eduACDC**: the notebook needs an internet connection the first time it runs in Colab, or when it is not opened from the `examples` folder.
- **Everything seems stuck or confused**: restart and run everything again (Colab: *Runtime → Restart session and run all*; JupyterLab: *Kernel → Restart Kernel and Run All Cells*).
