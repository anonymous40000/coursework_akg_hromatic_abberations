# pipeline.py: порядок стадий конвейера (фрагмент)
    source = frame_bgr.copy()
    before_geometry = residual(source)
    geometry_image, geometry_report = correct_auto(source)
    after_geometry = residual(geometry_image)

    working = geometry_image
    axial_report = None
    if enable_axial:
        kernel_b, kernel_r = axial_kernels if axial_kernels else (None, None)
        working, axial_report = correct_axial(
            geometry_image, kernel_b=kernel_b, kernel_r=kernel_r
        )

    if enable_defringe:
        final_image, fringe_report = defringe(working)
