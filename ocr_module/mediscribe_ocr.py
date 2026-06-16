from ocr_module.prescription_ocr import process_prescription as actual_perform_ocr


def perform_ocr(image_path, output_dir=None):
    return actual_perform_ocr(image_path, output_dir)
