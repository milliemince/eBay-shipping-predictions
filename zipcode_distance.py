import mpu
from uszipcode import SearchEngine
import pandas as pd


def get_distance(item_zip, buyer_zip):
    """
    Haversine formula using 'mpu' library which determines the
    great-circle distance between two points on a sphere.
    """
    if item_zip is not None and buyer_zip is not None:
        # print('item_zip: ' + item_zip)
        # print('buyer_zip: ' + buyer_zip)

        #for extensive list of zipcodes, set simple_zipcode =False
        search = SearchEngine(simple_zipcode=True)

        zip1 = search.by_zipcode(item_zip)
        lat1 =zip1.lat
        # print('zip1lat1: '+ str(lat1))
        long1 =zip1.lng
        # print('zip1lng1: '+ str(long1))

        zip2 =search.by_zipcode(buyer_zip)
        lat2 =zip2.lat
        # print('zip2lat2: '+ str(lat2))
        long2 =zip2.lng
        # print('zip2lng2: '+ str(long2))

        if lat1 is None or lat2 is None or long1 is None or long2 is None:
            return None

        return mpu.haversine_distance((lat1,long1),(lat2,long2))
    else:
        return None

def add_zip_distance_column(file_name):
    df = pd.read_csv(file_name)
    df['item_zip_str'] = df['item_zip'].apply(lambda x: str(x))
    df['buyer_zip_str'] = df['buyer_zip'].apply(lambda x: str(x))
    df['zip_distance'] = df.apply(lambda x: get_distance(x.item_zip_str, x.buyer_zip_str), axis=1)
    del df['buyer_zip_str']
    del df['item_zip_str']
    pd.DataFrame.to_csv(df, 'test_distance.csv')

if __name__ == "__main__":
    file_name = ('test.csv')
    add_zip_distance_column('eBay_ML_Challenge_Dataset_2021/eBay_ML_Challenge_Dataset_2021_train.tsv')
    # get_distance('04007', '49036')
