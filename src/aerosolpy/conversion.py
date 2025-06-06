################################################################
# aim of this script is to have a class that                   #
# converts the one minute raw files zipped                     #
# into json or csv files. Two modes of working:                #
# whether used by the machine to produce the last              #
# particle, or used by the user to convert a full              #
# directory.                                                   #
# -------------------------------------------------------------#
# +INPUT: - directory with zipfiles                            #
#         - filename to convert                                #
#         - filename to save                                   #
#         - output format: json, csv                           #
#         - keep/not the threshold                             #
#                       (default= false)                       #
#         - mode='last' or 'user'                              #
#                                                              #
# +OUTPUT:                                                     #
#         - directory with json/csv files                      #
#                                                              #
# +USAGE: -python3 conversion.py -d "/path/" -o "csv" -m 'user'#
#         -python3 conversion.py -d "/path/" -o "csv" -t       #
#         -python3 conversion.py -f "file" -s "save/file"      #
#         -python3 conversion.py -h                            #
################################################################
import argparse
import datetime
import json
import os
import sys
import zipfile
import zlib
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file1", help="file to convert", default='', required=False)
    parser.add_argument("-d", "--directory", help="directory to make the conversion",
                        default=os.getcwd(), required=False)
    parser.add_argument("-s", "--save", help="file to be saved", default='', required=False)
    parser.add_argument("-o", "--output", help="possible choices of conversion: csv, json",
                        default='json', required=False)
    parser.add_argument("-t", "--threshold", action='store_true', help="save the threshold value",
                        default=False, required=False)
    parser.add_argument("-m", "--mode", help='mode to be used', default='last', required=False)
    parser.add_argument("-e", "--extra", help="size and asymmetry factor are saved", default=True, required=False)
    parser.add_argument("-z", '--zip', help="whether to keep the converted zip files or not", action="store_false",
                        default=True, required=False)
    return parser.parse_args()


class Conversion:
    def __init__(self, filename: str, file_save: str = '', output: str = 'json', extra_param: bool = True,
                 keep_threshold: bool = False, mode: str = 'last', scattering: List[float] = None,
                 spectrum: List[float] = None, lifetime: List[List] = None, keep_zip: bool = True):
        self.scattering = scattering if scattering is not None else np.linspace(-45, 45, 24).tolist()
        self.lifetime = lifetime if lifetime is not None else [[350, 400], [420, 460], [511, 572], [672, 800]]
        self.spectrum = spectrum if spectrum is not None else np.linspace(350, 800, 32).tolist()
        self.filename = filename  # these are the zip files
        self.directory = os.path.dirname(filename) + '/'
        self.output = output
        self.extra_param = extra_param
        self.keep_threshold = keep_threshold
        self.mode = mode
        self.keep_zip = keep_zip
        if self.mode != 'last':
            self.file_save = ''
        else:
            self.file_save = file_save

    def unzip_file(self):

        try:
            with zipfile.ZipFile(self.filename, "r") as zip_ref:
                name = zip_ref.namelist()[0]
                zip_ref.extractall(self.directory)

            if self.filename == self.directory + name[:-4] + '.zip':
                if not self.keep_zip:
                    os.remove(self.filename)

            else:
                os.rename(self.directory + name[:-4] + '.raw', self.filename[:-4] + '.raw')
                if not self.keep_zip:
                    os.remove(self.filename)
            return 0

        except:
            print("    Error in unziping the file %s" % self.filename)
            return -1

    def read_raw(self):
        if self.mode != 'last':
            self.unzip_file()
            f = open(self.filename[:-4] + '.raw', "rb")
        else:
            f = open(self.filename, "rb")

        detect_in_file = 0
        detect_total = 0
        nb_errors = 0
        data_vec = []

        while True:
            read_byte = f.read(4)
            if read_byte == b"":
                # print("End of file %s, number of detection = %d" %(f, detect_in_file))
                # print("#######################")
                detect_in_file = 0
                break

            header_fixed_code = int.from_bytes(read_byte, byteorder='big')
            if header_fixed_code != 0x22d5ebe7:
                print("*** Error header fixed code: %s - 0x22d5ebe7" % hex(header_fixed_code))
                nb_errors += 1
                break
            detect_in_file += 1
            detect_total += 1

            # General header
            self.version = int.from_bytes(f.read(4), byteorder='big')
            self.serial = int.from_bytes(f.read(4), byteorder='big')
            unix_time_seconds = int.from_bytes(f.read(4), byteorder='big')
            unix_time_ms = int.from_bytes(f.read(4), byteorder='big')
            time_stamp_conv = datetime.datetime.fromtimestamp(unix_time_seconds).strftime(
                '%Y-%m-%d %H:%M:%S')
            self.timestamp = ("%s.%s" % (time_stamp_conv, str(unix_time_ms)))
            number_of_modules = int.from_bytes(f.read(4), byteorder='big')
            f.read(12)

            # Scattering header
            lt11_framelength = int.from_bytes(f.read(4), byteorder='big')
            emitter_id = int.from_bytes(f.read(2), byteorder='big')
            detection_id = int.from_bytes(f.read(2), byteorder='big')
            image_size = int.from_bytes(f.read(4), byteorder='big')
            f.read(20)

            if number_of_modules == 3:
                # FluoHeader
                lt03_framelength = int.from_bytes(f.read(4), byteorder='big')
                emitter_id = int.from_bytes(f.read(2), byteorder='big')
                detection_id = int.from_bytes(f.read(2), byteorder='big')
                f.read(24)
                if lt03_framelength != 256:
                    print("***ERROR LT3 Header -- Wrong Frame length: %d" % lt03_framelength)
                    nb_errors += 1

                # LifeTimeHeader
                lt04_framelength = int.from_bytes(f.read(4), byteorder='big')
                emitter_id = int.from_bytes(f.read(2), byteorder='big')
                detection_id = int.from_bytes(f.read(2), byteorder='big')
                f.read(24)
                if lt04_framelength != 128:
                    print("***ERROR LT4 Header -- Wrong Frame length: %d" % lt04_framelength)
                    nb_errors += 1

            # Scattering Image, uint8, big-endian
            image_raw = f.read(image_size * 4)
            self.scattering_image = np.fromstring(image_raw, dtype=np.dtype('>u4')).tolist()  # add >>
            crc = zlib.crc32(image_raw)

            # Scattering Threshold, uint32 , big-endian
            thresholds = f.read((lt11_framelength - image_size) * 4)
            self.thresholds = np.fromstring(thresholds, dtype=np.dtype('>u4')).tolist()
            crc = zlib.crc32(thresholds, crc)

            if number_of_modules == 3:
                # Fluo Image, int32, big endian
                fluo_raw = f.read(lt03_framelength * 4)
                self.spect_image = np.fromstring(fluo_raw, dtype=np.dtype('>i4')).tolist()
                crc = zlib.crc32(fluo_raw, crc)

                # Lifetime Image, int16, big endian
                life_raw = f.read(lt04_framelength * 4)
                self.life_image = np.fromstring(life_raw, dtype=np.dtype('>i2')).tolist()
                crc = zlib.crc32(life_raw, crc)

            # Footer
            footer_crc = int.from_bytes(f.read(4), byteorder='big')
            if crc != footer_crc:
                print("***ERROR CRC do not match: %s - %s" % (hex(footer_crc), hex(crc)))
                nb_errors += 1
                break

            f.read(124)

            # check correct footer final code
            footer_fixed_code = int.from_bytes(f.read(4), byteorder='big')
            if footer_fixed_code != 0xf82f5be4:
                print("***Error footer fixed code: %s - 0xf82f5be4" % hex(footer_fixed_code))
                nb_errors += 1

            # appending particle to DIC
            data_vec.append(self.particles_to_dic(number_of_modules))

        return data_vec, self.serial, self.version

    @staticmethod
    def size_particle(image):
        x = np.asarray(image).sum()
        return 9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00

    @staticmethod
    def get_asymmetry(train):
        if len(train) < 3600:
            train_arr = np.asarray(train)
            train_ex = np.pad(train_arr, (0, 3600 - len(train_arr)), 'constant').reshape(-1, 24)

        else:
            train_ex = np.asarray(train[0:3600]).reshape(-1, 24)

        extra_zeroes = np.zeros([40, 24])
        train_ex = np.vstack([extra_zeroes, train_ex, extra_zeroes])
        sumit = np.sum(train_ex, axis=1)
        x = np.linspace(0, 230, 230)
        p = int(np.dot(sumit, x) / np.sum(sumit))
        train_tosend = train_ex[p - 40:p + 40].reshape(1, -1).tolist()[0]
        train_left = sorted(train_tosend[:40 * 24])
        train_right = train_tosend[40 * 24:]
        train_right_reverted = sorted(train_right[::-1])
        score = np.clip(np.abs(r2_score(train_left, train_right_reverted)), a_max=1, a_min=None)  # clip score at 1

        return 1 - score

    def global_dic(self):
        data_vec, serial, version = self.read_raw()
        dic = defaultdict(dict)
        dic['Header']['Device'] = {}
        dic['Header']['Device']['Serial'] = serial
        dic['Header']['Device']['Version'] = version
        dic['Header']['Alignment'] = {}
        dic['Header']['Alignment']['Scattering angles'] = self.scattering
        dic['Header']['Alignment']['Spectral wavelengths'] = self.spectrum
        dic['Header']['Alignment']['Lifetime ranges'] = self.lifetime
        dic['Header']['Units'] = {}
        dic['Header']['Units']['Lifetime'] = ['Time, ns', 'Amplitude, NA']
        dic['Header']['Units']['Spectrometer'] = ['Wavelength, nm', 'Amplitude, NA']
        dic['Header']['Units']['Scattering'] = ['Time, us', 'Angle, deg', 'Amplitude, NA']
        dic['Data'] = data_vec
        return dic

    def particles_to_dic(self, n_modules):
        gdic = defaultdict(dict)
        gdic['Timestamp'] = self.timestamp
        gdic['Scattering'] = {}
        gdic['Scattering']['Image'] = self.scattering_image
        if self.extra_param:
            gdic['Size'] = self.size_particle(self.scattering_image)
            gdic['TimeAsymmetry'] = self.get_asymmetry(self.scattering_image)
        if self.keep_threshold:  # write the scat thresholds if keep_thresholds true
            gdic['Scattering']['Thresholds'] = self.thresholds
        if n_modules == 3:
            gdic['Spectrometer'] = self.spect_image
            gdic['Lifetime'] = self.life_image
        return gdic

    def define_name(self):
        # exactly as raw_filename with json extension
        if self.output == 'json' and self.mode == 'user':
            return self.filename[:-4] + '.json'
        elif self.output == 'json' and self.mode == 'last':
            return self.file_save
        elif self.output == 'csv':
            return self.filename[:-4] + '.csv'
        else:
            print('Please, provide a valid choice among "json" or "csv"')
            sys.exit()

    def save_json(self):
        try:
            # print(self.define_name())
            with open(self.define_name(), 'w') as outfile:
                json.dump(self.global_dic(), outfile, indent=4, sort_keys=True,
                          separators=(',', ':'))
            return 0
        except:
            print("  Error writing the json file %s ..." % self.define_name())
            return -1

    def save_csv(self):
        data = self.global_dic()['Data']
        df = pd.DataFrame(data)
        if 'Lifetime' in df.columns:
            lifetime = pd.DataFrame(df.Lifetime.values.tolist())
            spectrometer = pd.DataFrame(df.Spectrometer.values.tolist())
        scattering = pd.DataFrame(df.Scattering.values.tolist())
        scatter_image = pd.DataFrame(scattering.Image.values.tolist()).fillna(0)
        if 'Lifetime' in df.columns:
            df_final = pd.concat([lifetime, spectrometer, scatter_image], axis=1)
        else:
            df_final = pd.concat([scatter_image], axis=1)

        if self.keep_threshold:
            thresh = pd.DataFrame(scattering.Thresholds.values.tolist())
            df_final = pd.concat([df_final, thresh], axis=1)

        df_final.to_csv(self.filename[:-4] + '.csv', index=False, header=None)  # save into csv
        if 'Lifetime' in df.columns:
            del df, lifetime, spectrometer, scattering, scatter_image, df_final  # release RAM
        else:
            del df, scattering, scatter_image, df_final  # release RAM
        return 0

    def save_overall(self):
        if self.output == 'json':
            self.save_json()
        if self.output == 'csv':
            self.save_csv()

    def remove_file(self):
        try:
            os.remove(self.filename[:-4] + '.raw')
            return 0
        except:
            print("Cannot remove filename %s" % self.filename[:-4] + '.raw')
            return -1


if __name__ == '__main__':

    args = parse()

    if args.mode != 'last':
        if os.path.isdir(args.directory):
            for filename in tqdm(os.listdir(args.directory)):
                converter = Conversion(args.directory + filename, args.save,
                                       output=args.output, keep_threshold=args.threshold,
                                       mode=args.mode, keep_zip=args.zip)

                converter.save_overall()
                converter.remove_file()
            print('')
            print('Conversion finished!')
            print('-' * 60)
        else:
            print("Directory %s doesn't exist" % args.directory)
            sys.exit()

    else:
        converter = Conversion(args.file1, args.save)
        converter.save_overall()
