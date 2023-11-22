filelist = getArgument()
print("Importing .TIFF sequence:")
print(filelist)
run("Image Sequence...", "open=" + filelist + " sort use");
print("Done.")