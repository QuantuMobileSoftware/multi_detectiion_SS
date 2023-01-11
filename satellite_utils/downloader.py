from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from subprocess import call
import os
from pathlib import Path
import zipfile
import requests
import geojson
from requests.auth import HTTPBasicAuth
import shutil
import sys
import time
from tempfile import TemporaryDirectory
import pandas as pd
from typing import Tuple
import re
import unicodedata

from .constants.constants import assets_in_bundles_for_visualizing
bundles_json = 'constants/bundles.json'

class PlanetAuth:
    def __init__(self, auth_key_path):
        """
        @param auth_key_path: absolute path to json file with api_key value
        """
        self.auth = self.get_auth(auth_key_path)
    
    @staticmethod
    def get_auth(auth_key_path):
        with open(auth_key_path, 'r') as f:
            return HTTPBasicAuth(json.load(f)['api_key'], '')


class PlanetOrderDownloader():
    orders_url = 'https://api.planet.com/compute/ops/orders/v2'
    
    def __init__(self, auth_key, download_path):
        """
        @param auth_key: str or Path:  api_key value
        @param download_path: str or Path: absolute path for zip archive downloading
        """
        self.auth = HTTPBasicAuth(auth_key, '')
        self.order_state = None
        self.order_id = None
        self.order_url = None
        self.order_name = None
        self.order_archive_url = None
        self.order_archive_name = None
        self.manifest_url = None
        self.order_archive_size = None
        self.order_archive_digests = None
        self.base_path=Path(download_path)

    def set_order_id(self, order_id):
        self.order_id = order_id
        self.order_url = self.get_order_url()
    
    def get_order_url(self):
        return f'{self.orders_url}/{self.order_id}'

    @staticmethod
    def slugify(value):
        """
        Convert to ASCII. Convert spaces to hyphens.
        Remove characters that aren't alphanumerics, underscores, or hyphens.
        Convert to lowercase. Also strip leading and trailing whitespace.
        """
        value = str(value)
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        value = re.sub(r'[^\w\s-]', '', value.lower()).strip()
        return re.sub(r'[-\s]+', '-', value)
    
    def dump_no_valid_geosjon(self, polygon, geojson_path):
        label = 'Invalid orderid'
        style = dict(color='red')
        feature = geojson.Feature(geometry=polygon, properties=dict(label=label, style=style))
        response_folder = geojson_path / self.order_id
        response_folder.mkdir(parents=True, exist_ok=True)
        with open(response_folder / 'response.geojson', 'w') as f:
            geojson.dump(feature, f)

    def poll_for_success(self, timeout=10):
        """
        Waiting for 'success' or 'partial' response state for order
        @param timeout: int: timeout between requests
        @return:
        """
        while True:
            r = requests.get(self.order_url, auth = self.auth)
            response = r.json()
            if r.status_code == 404:
                print(f'Given ORDERID - {self.order_id} is not valid.')
                sys.exit(1)
            if 'code' in response.keys():
                if response['code'] == 601:
                    print(response['message'])
                    sys.exit(1)
            state = response['state']
            if state == 'success' or state == 'partial':
                self.order_state = state
                print(f'Order {self.order_id} is {self.order_state}.')
                break
            if state == 'failed':
                print(f'Order {self.order_id} is {self.order_state}.')
                sys.exit(1)
            time.sleep(timeout)
            
    def check_expires_date(self, expires_date):
        """
        Compare datetime.now() with expires_date.
        If expires_date < datetime.now() exit the script.
        :param expires_date: str: datatime with milliseconds, example - '2022-04-13T07:40:06.907Z'
        :return:
        """
        expires_date = time.strptime(expires_date, '%Y-%m-%dT%H:%M:%S.%fZ')
        if expires_date < time.strptime(str(datetime.now()), '%Y-%m-%d %H:%M:%S.%f'):
            print(
                f"""
                order {self.order_name} with id {self.order_id} is expired!
                You can submit a request for inquiries about orders placed more than three months ago!
                https://support.planet.com/hc/en-us/requests/new/
                """
            )
            sys.exit(1)
            
    def download_order_info(self, n_requests=10):
        """
        Get order info and save it as 'results.json' file.
        If requests during n_requests were unsuccessful exit the script.
        @param n_requests: int: number of requests we will try to get order results.
        @return: json: order info
        """
        for i in range(0, n_requests):
            print(f'GET  {self.order_url}')
            response = requests.get(self.order_url, auth=self.auth)
            print(f'GET {self.order_url} returned {response.status_code} status code!')
            if response.status_code == 200:
                data = response.json()
                with open(self.download_path / 'results.json', 'w') as f:
                    f.write(json.dumps(data))
                return data
            time.sleep(10)
        print(f'GET {self.order_url} was unsuccessful!')
        sys.exit(1)
        
    def download_archive_info(self, n_requests=10):
        """
        Get archive info and save it as 'manifest.json' file.
        If requests during n_requests were unsuccessful exit the script.
        @param n_requests: int: number of requests we will try to get archive info.
        @return: json
        """
        for i in range(0, n_requests):
            print(f'GET  {self.manifest_url}')
            response = requests.get(self.manifest_url, auth=self.auth)
            print(f'GET {self.manifest_url} returned {response.status_code} status code!')
            if response.status_code == 200:
                data = response.json()
                with open(self.download_path / 'manifest.json', 'w') as f:
                    f.write(json.dumps(data))
                return data
            time.sleep(10)
        print(f'GET {self.manifest_url} was unsuccessful!')
        sys.exit(1)

    def get_order_info(self):
        """
        Get and extract order name and order archive name.
        @return: order_name: str, order_archive_name: str
        """
        self.download_path = self.base_path / self.order_id
        self.download_path.mkdir(parents=True, exist_ok=True)
        if (self.download_path / 'results.json').exists():
            with open(self.download_path / 'results.json', 'r') as f:
                data = json.load(f)
        else:
            data = self.download_order_info()
        self.order_name = data['name']
        results = data['_links']['results']
        for result in results:
            # self.check_expires_date(result['expires_at'])
            if Path(result['name']).suffix == '.json':
                self.manifest_url = result['location']
                if (self.download_path / 'manifest.json').exists():
                    with open(self.download_path / 'manifest.json', 'r') as f:
                        archive_info = json.load(f)
                else:
                    archive_info = self.download_archive_info()
                self.order_archive_size = archive_info['files'][0]['size']
                self.order_archive_digests = archive_info['files'][0]['digests']
            if Path(result['name']).suffix == '.zip':
                self.order_archive_url = result['location']
                self.order_archive_name = Path(result['name']).name
        return self.order_name, self.order_archive_name
        
    def download_order_archive(self, n_requests=10):
        """
        Download order archive.
        @param n_requests: int: number of requests we will try to get archive info.
        @return: None
        """
        archive_path = self.download_path / self.order_archive_name
        if not archive_path.exists():
            with open(archive_path, 'w') as f:
                f.write('')
        print('downloaded size:', archive_path.stat().st_size)
        print('order_archive_size:', self.order_archive_size)
        print(f'downloading {self.order_archive_name} to {archive_path}')
        chunk_length = 16*1024*1024
        
        while n_requests > 0:
            if self.order_archive_size - archive_path.stat().st_size == 0:
                break
            headers = {"Range": f"bytes={archive_path.stat().st_size}-"}
            response = requests.get(self.order_archive_url, stream=True, headers=headers)
            if response.status_code == 206:
                with open(archive_path, 'ab') as f:
                    shutil.copyfileobj(response.raw, f, length=chunk_length)
            else:
                print(f'GET {self.order_archive_url} returned {response.status_code} status code!')
            n_requests = n_requests - 1
        
        if archive_path.stat().st_size < self.order_archive_size:
            print(f'GET {self.order_archive_url} during {n_requests} was unsuccessful! Aborting!')
            sys.exit(1)
            
        self.archive_path = archive_path
        self.get_products_info()
        self.planet_bundles = self.get_planet_bundles()
        self.extract_archive_manifest_data()
        self.extract_archive_item_json_files()
        print(f'File {archive_path} was downloaded')

    def extract_archive_manifest_data(self):
        """
        Extract manifest.json from archive,
        convert it to self.found_items_info_df Pandas DataFrame
        and self.found_products_df Pandas DataFrame
        :return:
        """
        with TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
                zip_ref.extract('manifest.json', path=tmp_dir)
            
            with open(Path(tmp_dir) / 'manifest.json') as f:
                archive_files_info = json.load(f)['files']
            with open(Path(tmp_dir) / 'manifest.json', 'w') as f:
                f.write(json.dumps(archive_files_info))
            _df = pd.read_json(Path(tmp_dir) / 'manifest.json',)
            self.found_items_info_df = self.filter_product_by_file_type(_df, 'application/json')
            self.found_products_df = self.filter_product_by_file_type(_df, 'image/tiff')
        return
    
    @staticmethod
    def filter_product_by_file_type(df, media_type):
        """
        Clear products dataframe from rows which contain other media_type
        :param media_type: str: 'image/tiff' or 'application/json'
        :param df: Pandas DataFrame
        :return: Pandas DataFrame
        """
        df = df.loc[df['media_type'] == media_type]
        df.reset_index(drop=True, inplace=True)
        return df
    
    def extract_archive_item_json_files(self):
        """
        Extract all '.json' files from order archive
        :return:
        """
        future_to_json_files_list = []
        start_time = time.time()
        max_workers = os.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, row in self.found_items_info_df.iterrows():
                file_path = row['path']
                file_size = row['size']
                future = executor.submit(self.extract_file_from_archive, file_path, file_size)
                future_to_json_files_list.append(future)
        print(f'{time.time() - start_time} seconds for extracting of all json files from archive')
    
    def get_products_info(self):
        """
        Extract data from 'results.json' file
        :return:
        """
        self.results_json_path = self.archive_path.parent / 'results.json'
        with open(self.results_json_path) as f:
            data = json.load(f)
            self.products_info = data['products']
            self.name = self.slugify(data['name'])
        return
    
    @staticmethod
    def get_planet_bundles():
        """
        Get planet bundles information from json file
        :return: obj: planet bundles information
        """
        with open(Path(__file__).parent.resolve() / bundles_json, 'r') as f:
            return json.load(f)

    def extract_archive_item_json_files(self):
        """
        Extract all '.json' files from order archive
        :return:
        """
        future_to_json_files_list = []
        start_time = time.time()
        max_workers = os.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, row in self.found_items_info_df.iterrows():
                file_path = row['path']
                file_size = row['size']
                future = executor.submit(self.extract_file_from_archive, file_path, file_size)
                future_to_json_files_list.append(future)
        print(f'{time.time() - start_time} seconds for extracting of all json files from archive')

    def extract_file_from_archive(self, file_path, file_size):
        path = self.base_path / file_path
        if path.exists() and path.stat().st_size == file_size:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            zip_ref.extract(file_path, path=self.base_path)
        return
    
    @staticmethod
    def get_asset_properties(product_bundle, item_type):
        """
        Check if product_bundle and item_type are in known bundles for visualizing,
        if it is not, then raise KeyError,
        else return dict with file name for extracting from archive and product properties
        :param product_bundle: str: product bundle name
        :param item_type: str: product item type
        :return: dict: {'file_name': 'ortho_visual', 'properties':
        {'bands': {'R': 1, 'G': 2, 'B': 3}, 'projection': 'UTM', 'dtype': 'uint8'}}
        """
        if product_bundle not in assets_in_bundles_for_visualizing.keys():
            print('Unknown planet/bundle_type!', product_bundle)
            raise KeyError
        if item_type not in assets_in_bundles_for_visualizing[product_bundle].keys():
            print('Unknown planet/asset_type! in assets_in_bundles_for_visualizing', item_type)
            raise KeyError
        return assets_in_bundles_for_visualizing[product_bundle][item_type]

    def merge_tiles(self, paths, out_raster_path='test.tif'):
        print(f'Start merging to {out_raster_path}')

        start_time = time.time()
        listToStr = ' '.join([str(elem) for elem in paths])
        call(' '.join(["gdalwarp --config GDAL_CACHEMAX 3000 -wm 3000 -t_srs EPSG:3857", listToStr, str(out_raster_path)]),
                    shell=True)
        
        print(f'{time.time() - start_time} seconds for merging {len(paths)} images in to {out_raster_path}')
        return out_raster_path
    
    @staticmethod
    def filter_product_by_bundle_assets_file_name(products_df, file_name):
        return products_df[products_df.annotations.apply(lambda row: row['planet/asset_type'] == file_name)]

    @staticmethod
    def get_empty_img_df():
        column_names = ['path', 'size', 'item_id', 'bands_order', 'dtype']
        return pd.DataFrame(columns=column_names)
    
    @staticmethod
    def filter_product_by_item_id(row, item_id):
        return row['planet/item_id'] == item_id
    
    def get_item_info(self, item_id, item_type):
        """
        Extract item 'metadata.json' file from archive and read it
        :param item_id: str: planet product item_id
        :param item_type: str: planet product item_type
        :return: obj or None: data from metadata.json for product item
        """
        item_metadata_path = self.base_path / 'files' / item_type / item_id / f'{item_id}_metadata.json'
        if not item_metadata_path.exists():
            print(f'{item_metadata_path} doth not exists! Skip it!')
            return
        with open(item_metadata_path) as f:
            item_info = json.load(f)
            item_info['path'] = item_metadata_path
        return item_info

    def product_filter(self, product):
        """
        Filter ordered images
        :param product: obj:
        :return: pandas DataFrame: DataFrame with column names: 'path', 'size', 'item_id', 'bands_order'
        """
        img_df = self.get_empty_img_df()
        for item_id in product['item_ids']:
            item_info = self.get_item_info(item_id, product['item_type'])
            product_bundle = product['product_bundle']
            item_type = item_info['properties']['item_type']
            item_to_process = self.filtered_products_df[self.filtered_products_df.annotations.apply(
                lambda row: self.filter_product_by_item_id(row, item_id)
            )]
            item_path_in_archive = item_to_process['path'].values[0]
            item_size = item_to_process['size'].values[0]
            bands_order = self.get_bands_order_for_visualization(product_bundle, item_type)
            asset_dtype = self.get_asset_dtype(product_bundle, item_type)
            
            # add new row to img_df
            img_df.loc[img_df.shape[0]] = [item_path_in_archive, item_size, item_id, bands_order, asset_dtype]
        return img_df
    
    @staticmethod
    def get_asset_dtype(product_bundle, item_type):
        return assets_in_bundles_for_visualizing[product_bundle][item_type]['properties']['dtype']
    
    @staticmethod
    def get_product_bands_order(product_bundle, item_type):
        """
        Get available bands for product based on product bundle and product item_type
        :param product_bundle: str: product bundle
        :param item_type: str: product item_type
        :return: Dict: bands dict where keys can be B', 'G', 'R', 'NIR', 'Red-Edge', 'Yellow', 'G_I', 'Coastal_Blue'
        """
        return assets_in_bundles_for_visualizing[product_bundle][item_type]['properties']['bands']
    
    def get_bands_order_for_visualization(self, product_bundle, item_type):
        """
        Extract from constants bands order for visualization based on product bundle and product item_type
        :param product_bundle: str: product bundle
        :param item_type: str: product item_type
        :return: list: list of ints
        """
        bands_to_extract = ['R', 'G', 'B']
        product_bands_order = self.get_product_bands_order(product_bundle, item_type)
        return [product_bands_order[band] for band in bands_to_extract]
    
    def extract_img_files_from_archiwe(self, img_df):
        future_to_img_files_list = []
        start_time = time.time()
        max_workers = os.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, row in img_df.iterrows():
                file_path = row['path']
                file_size = row['size']
                future = executor.submit(self.extract_file_from_archive, file_path, file_size)
                future_to_img_files_list.append(future)
        print(f'{time.time() - start_time} seconds for extracting of all product files from archive')

    @staticmethod
    def change_raster_dtype(raster_path, dtype="Int16") -> str:
        """Convert raster's pixel dtype into given

        Args:
            raster_path (_type_): Path to raster to convert
            dtype (str, optional): Output raster dtype. Defaults to "Int16".

        Returns:
            str: Path to raster with new dtype 
        """

        pth = Path(raster_path)
        output_raster = os.path.join(pth.parent, pth.stem+f"_{dtype}.tif")
        call(f"gdal_translate -ot Int16 {raster_path} {output_raster}",shell=True)
        return output_raster


    def run(self) -> Tuple[str, list]:
        """Create suitable for prediction raster:
        Merge multiple rasters into single (it there are mote then one) and pixel dtype = int16.
        Standard Planet's Uint16 pixel's dtype is not suitable for torch.

        Returns:
            Tuple[str, list]: Path to raster for prediction and RGB bands order
        """
        for product in self.products_info:
            item_type = product['item_type']
            product_bundle = product['product_bundle']
            asset_properties = self.get_asset_properties(product_bundle, item_type)
            bands_order = self.get_bands_order_for_visualization(product_bundle, item_type)

            self.filtered_products_df = self.filter_product_by_bundle_assets_file_name(
                self.found_products_df, asset_properties['file_name']
            )
            self.img_df = self.product_filter(product)    
            self.extract_img_files_from_archiwe(self.img_df)

            image_list = [self.base_path / pth  for pth in self.img_df.path.to_list()]
            
            if len(image_list) > 1:
                merged_raster_path = self.merge_tiles(
                    image_list,
                    self.base_path / Path(self.name).with_suffix('.tif')
                )
            else:
                merged_raster_path = image_list[0]

            int16_raster = self.change_raster_dtype(merged_raster_path)
            return int16_raster, bands_order
