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
        # code to iterate over files in a directory to compute patch mean and std for noise profiling

        # for temporal noise estimation, set the raw_Path to frame differnces dir otherwise black level corrected frames dir
        raw_path = Path("/home/user3/Desktop/Maria Nadeem/Sensor Noise synthesis/IMX678/2frames_at_50exposures/white_light_100%/IMX678_27/BurstCapture_Pairs/single_frame")
        raw_files  = [file_path for file_path in raw_path.iterdir() if file_path.suffix == '.raw']

        # sort the files in increasing exposure levels (file numbering: 50, 49,48...1) 
        raw_files.sort(reverse=True)

        for raw_path in raw_files:
            # print(f"Processing file: {raw_path.name}")
            self.raw_image_para.file_name = raw_path.name
            raw_image = np.fromfile(raw_path, dtype=np.uint16).reshape((self.raw_image_para.height, self.raw_image_para.width))
            self.raw_image_para.file_name = raw_path.name
            yield raw_image
    
    def save_csv_files(self, data_dict, column_name_tag, output_dir):
        """
        Save a CSV file for the provided dictionary.
        Each CSV file corresponds to the data in the dictionary, and the file name is based on the column name tag.
        """
        # output_dir = Path("/home/user3/Desktop/Maria Nadeem/Sensor Noise synthesis/IMX678_138/PTC_data_rewritecsv")
        output_dir.mkdir(parents=True, exist_ok=True)

        first_key = next(iter(data_dict))
        num_rows = len(data_dict[first_key])
        
        for row_idx in range(num_rows):

            file_path = output_dir / f"{output_dir.parent.parent.name}_no_normalization_{column_name_tag}_patch{row_idx + 1}.csv"
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # Write the header row
                # writer.writerow(["Image_Name", f"R_{column_name_tag}", f"G_{column_name_tag}", f"B_{column_name_tag}"])
                writer.writerow(["Image_Name", f"{column_name_tag}"])


                for image_name, data_matrices in data_dict.items():
                    row = data_matrices[row_idx]
                    writer.writerow(
                            [image_name, f"{row[0]:.6f}"])
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
        temp_std = {}
        output_dir = Path("/home/user3/Desktop/Maria Nadeem/Sensor Noise synthesis/IMX678/2frames_at_50exposures/white_light_100%/IMX678_27/BurstCapture_Pairs/PTC_data_singleframe")
        # set the temp_noise_std to False for mean and variance calculation on black
        # level corrected frames
        temp_noise_flag = False

        for raw_image in raw_gen:
            self.raw_image_para.raw_image = raw_image
            
            
            var_mat, mean_mat, temp_noise_std = noise_est.apply_algo(temp_noise_std=temp_noise_flag)
            variances[self.raw_image_para.file_name] = var_mat
            means[self.raw_image_para.file_name] = mean_mat
            
            if temp_noise_std is not None:
                temp_std[self.raw_image_para.file_name] = temp_noise_std

            # print(f"completed NE for: {self.raw_image_para.file_name}")
        if temp_std:
            self.save_csv_files(temp_std, "temporal_noise_SD", output_dir)
        else:
            self.save_csv_files(variances, "variance", output_dir)
            self.save_csv_files(means, "mean", output_dir)
        

        generate_separator("Noise Levels Estimated Successfully!", "-")
        generate_separator("", "*")
        return True
