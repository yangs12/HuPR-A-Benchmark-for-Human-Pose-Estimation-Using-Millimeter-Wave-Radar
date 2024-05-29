import os
import numpy as np
import pandas as pd

folder_path = "/home/jupyter/hupr/radar_processed/"

# Get the list of files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file
file_exists = []
# for file_name in file_list:
for file_idx in range(200,250):
    if file_idx!=230:
        file_name = f'single_{file_idx}'
        # if file_name.startswith('single'):
        # if file_name not in ['single_52','single_201','single_55','single_123',
        # 'single_6', "single_211", "single_89", "single_192", "single_150", "single_164",
        # "single_66", "single_95", "single_153", "single_140", "single_149",
        # "single_40", "single_214", "single_271", "single_221", "single_110",
        # "single_171", "single_251", "single_71", "single_231", "single_188",
        # "single_179", "single_193", "single_91", "single_257", "single_239",
        # "single_27", "single_141", "single_85", "single_264", "single_69",
        # "single_216", "single_134", "single_212", "single_165",'single_25','single_29',"single_100",
        # "single_90", "single_92", "single_93", "single_94", "single_96", "single_97", "single_98",
        # "single_99", "single_101", "single_102", "single_103", "single_104", "single_105",
        # "single_106", "single_107", "single_108", "single_109", "single_111", "single_112",
        # "single_113", "single_114", "single_115", "single_116", "single_117", "single_118",
        # "single_119", "single_120", "single_121", "single_122", "single_124", "single_125",
        # "single_126", "single_127", "single_128", "single_129", "single_130", "single_131",
        # "single_132", "single_133", "single_135", "single_136", "single_137", "single_138",
        # "single_139", "single_142", "single_143", "single_144", "single_145", "single_146",
        # "single_147", "single_148","single_1", "single_2", "single_3", "single_4", "single_5", "single_7", "single_8",
        # "single_9", "single_10", "single_11", "single_12", "single_13", "single_14", "single_15",
        # "single_16", "single_17", "single_18", "single_19", "single_20", "single_21", "single_22",
        # "single_23", "single_24", "single_26", "single_28", "single_30", "single_31", "single_32",
        # "single_33", "single_34", "single_35", "single_36", "single_37", "single_38", "single_39",
        # "single_41", "single_42", "single_43", "single_44", "single_45", "single_46", "single_47",
        # "single_48", "single_49", "single_50", "single_51", "single_53", "single_54", "single_56",
        # "single_57", "single_58", "single_59", "single_60", "single_61", "single_62", "single_63",
        # "single_64", "single_65", "single_67", "single_68", "single_70", "single_72", "single_73",
        # "single_74", "single_75", "single_76", "single_77", "single_78", "single_79", "single_80",
        # "single_81", "single_82", "single_83", "single_84", "single_86", "single_87", "single_88","single_151", "single_152", "single_154", "single_155", "single_156", "single_157",
        # "single_158", "single_159", "single_160", "single_161", "single_162", "single_163",
        # "single_166", "single_167", "single_168", "single_169", "single_170", "single_172",
        # "single_173", "single_174", "single_175", "single_176", "single_177", "single_178",
        # "single_180", "single_181", "single_182", "single_183", "single_184", "single_185",
        # "single_186", "single_187", "single_189", "single_190", "single_191", "single_194",
        # "single_195", "single_196", "single_197", "single_198", "single_199",]:
            # if file_name in ['single_44','single_29']:
        file_path = os.path.join(folder_path, file_name)
        print(file_name)
        for radar in ['/hori', '/vert']:

            for i in range(600):
            # print(i)
            # print(file_path+radar+f'/{i:09d}.npy')
                data = np.load(file_path+radar+f'/{i:09d}.npy', allow_pickle=True)
                shape = data.shape
                if np.isnan(data).any():
                    print(f"NaN values found in file: {file_name}")
                    file_exists.append((file_name))
                    # # file_exists.append((file_name, shape))
                    # if shape != (16,64,64,8):
                    #     print(file_name, shape)
                    #     file_exists.append((file_name, shape))
# np.save('shape_errors.npy', file_exists)
# Convert the list to a DataFrame
df = pd.DataFrame(file_exists, columns=['File Name'])

# Save the DataFrame to an Excel file
df.to_excel('/home/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/nan_errors.xlsx', index=False)