"""
File: bnr_module.py
Description: Executes the module flow with the algorithm for the bayer noise estimation
Author: 10xEngineers
------------------------------------------------------------
"""
from src.modules.BNR.bnr_algo import BneAlgo as bne_algo
from src.utils.algo_common_utils import select_image_and_get_para, generate_separator
from src.utils.area_selection_frame import SelectAreaFrame as select_area_frame
from pathlib import Path
import numpy as np
import csv

class BneModule:
    """
    Bayer Noise Estimation Module
    """

    def __init__(self):
        self.raw_image_para = None
        self.selection_frame = None

    def is_image_and_para_loaded(self):
        """
        To check if the raw image is loaded, if true store respective parameters.
        """
        file_type = (("RAW Files", "*.raw"),)

        is_selected, self.raw_image_para = select_image_and_get_para(file_type)

        return is_selected

    def color_checker_selection_frame(self):
        """
        Open the color checker patches selection frame and return true
        if patches are drawn and saved using continue button otherwise
        return false
        """
        self.selection_frame = select_area_frame(self.raw_image_para.rgb_image)

        if self.selection_frame.data.is_data_saved is False:
            return False
        return True

    def iterate_over_dir(self):
        # code to iterate over files in a directory to compute patch mean nad std for noise profiling

        raw_path = Path("/home/user3/Desktop/Maria Nadeem/Sensor Noise synthesis/IMX678_9/offset_corrected_frames")
        raw_files  = [file_path for file_path in raw_path.iterdir() if file_path.suffix == '.raw']

        # sort the files in increasing exposure levels (file numbering: 50, 49,48...1) 
        raw_files.sort(reverse=True)

        for raw_path in raw_files:
            # print(f"Processing file: {raw_path.name}")
            self.raw_image_para.file_name = raw_path.name
            raw_image = np.fromfile(raw_path, dtype=np.uint16).reshape((self.raw_image_para.height, self.raw_image_para.width))
            self.raw_image_para.file_name = raw_path.name
            yield raw_image
    
    def save_csv_files(self, variances, means):
        """
        Save multiple CSV files for each row of the 6x3 matrices in the provided dictionaries.
        Each CSV file corresponds to a row of the matrix, and the file name is based on the patch number.
        """
        output_dir = Path("/home/user3/Desktop/Maria Nadeem/Sensor Noise synthesis/IMX678_9/PTC_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get the number of rows from any one of the matrices (assuming all have the same shape)
        first_key = next(iter(variances))
        num_rows = len(variances[first_key])

        for row_idx in range(num_rows):
            file_path = output_dir / f"{output_dir.parent.name}_ptc_data_patch{row_idx + 1}.csv"
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Image_Name", "R_mean", "G_mean", "B_mean", "R_var", "G_var", "B_var"])
                for image_name, var_matrix in variances.items():
                    mean_matrix = means.get(image_name)
                    if mean_matrix is None:
                        print(f"Warning: No mean matrix found for {image_name}. Skipping.")
                        continue
                    # Get the row corresponding to the current patch
                    var_row = var_matrix[row_idx]
                    mean_row = mean_matrix[row_idx]
                    writer.writerow(
                        [image_name, 
                         f"{mean_row[0]:.6f}", f"{mean_row[1]:.6f}", f"{mean_row[2]:.6f}",
                         f"{var_row[0]:.6f}", f"{var_row[1]:.6f}", f"{var_row[2]:.6f}"]
                    )
                writer.writerow([])

        print(f"CSV file saved to:\n {file_path}")


    def implement_bne_algo(self):
        """
        Extract patches and apply algorithm on the image
        to estimate bayer noise levels.
        """
        # Extracting coordinates for the patches to be used
        # for extracting patches from each channel
        sub_rect_points = self.selection_frame.get_sub_rect_points()

        # Applying Noise Estimation Algorithm
        noise_est = bne_algo(self.raw_image_para, sub_rect_points)
        raw_gen = self.iterate_over_dir()

        variances = {}
        means = {}
        for raw_image in raw_gen:
            self.raw_image_para.raw_image = raw_image
            
            var_mat, mean_mat = noise_est.apply_algo()
            variances[self.raw_image_para.file_name] = var_mat
            means[self.raw_image_para.file_name] = mean_mat
            # print(f"completed NE for: {self.raw_image_para.file_name}")
        self.save_csv_files(variances, means)
        generate_separator("Noise Levels Estimated Successfully!", "-")
        generate_separator("", "*")
        return True
