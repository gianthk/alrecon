filelist = getArgument()
print("Importing .TIFF sequence:")
print(filelist)
// "TIFF Virtual Stack..." "Image Sequence..."
run("TIFF Virtual Stack...", "open=" + filelist + " sort use");
print("Done.")