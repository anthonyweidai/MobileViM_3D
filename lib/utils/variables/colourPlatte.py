import imgviz


COLOUR_CODES = {
    "twoclasses": [
        [0, 0, 0], # background
        [255, 255, 255]
    ],
    **dict.fromkeys(
        [
            "default", 
            "atlas", "toothfairy2", "pengwin", "brats2024t3",
        ], 
        list(map(list, imgviz.label_colormap(256)))
    ),
}