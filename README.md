# Move Multispectral Drone Photos for PPK Processing

This repository includes a script for preparing drone photo folders for PPK processing in Emlid Studio.

## Script

Use the following script in the `scripts` folder:

```text
scripts/move_MS_drone_photos_for_PPK.py
```

## What the script does

The script moves multispectral photos (`.tif` files) in each flight folder into a separate `MS` subfolder.

This leaves the RGB `.jpg` photos in the main flight folder, which allows Emlid Studio to perform PPK correction on the remaining RGB photos.

## How to use

Open `move_MS_drone_photos_for_PPK.py` and update the `ROOT_DIR` variable to the folder that contains your drone photo folders.

For example, change:

```python
ROOT_DIR = r"D:\SfM\20260326_DCC_photos"
```

to the path containing your drone photos:

```python
ROOT_DIR = r"YOUR\PATH\TO\DRONE\PHOTOS"
```

Then run the script.

## Expected folder result

Before running:

```text
Flight_Folder/
├── IMG_0001.jpg
├── IMG_0002.jpg
├── IMG_0001.tif
├── IMG_0002.tif
```

After running:

```text
Flight_Folder/
├── IMG_0001.jpg
├── IMG_0002.jpg
└── MS/
    ├── IMG_0001.tif
    └── IMG_0002.tif
```
