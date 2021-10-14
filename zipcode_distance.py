import mpu
from uszipcode import SearchEngine
import pandas as pd
import geopy.distance


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
        # print('city: ' + str(zip1.common_city_list) + ', '+ zip1.state)
        #print('zip 1: ' + str(zip1))
        lat1 =zip1.lat
        #print('zip1lat1: '+ str(lat1))
        long1 =zip1.lng
        #print('zip1lng1: '+ str(long1))

        zip2 =search.by_zipcode(buyer_zip)
        #print('city: ' + str(zip2.common_city_list) + ', '+ zip2.state)
        #print('zip 2: ' + str(zip2))
        lat2 =zip2.lat
        #print('zip2lat2: '+ str(lat2))
        long2 =zip2.lng
        #print('zip2lng2: '+ str(long2))

        if lat1 is None or lat2 is None or long1 is None or long2 is None:
            return None
        #print('dist: ' + str(mpu.haversine_distance((lat1,long1),(lat2,long2))))

        # should output in kilometers
        return mpu.haversine_distance((lat1,long1),(lat2,long2)) #, geopy.distance.vincenty(c(lat1,long1), (lat2,long2)).km
    else:
        return None


def add_zip_distance_column(item_zip, buyer_zip):
    item_zip_str = item_zip.apply(lambda x: str(x))
    buyer_zip_str = buyer_zip.apply(lambda x: str(x))

    zips = pd.concat([item_zip_str, buyer_zip_str], axis=1)

    zips['distance'] = zips.apply(lambda x: get_distance(x.item_zip, x.buyer_zip), axis=1)
    
    #print(zips['distance'])
    return zips['distance']

if __name__ == "__main__":
    test_file = pd.read_csv('test.csv')
    # #print(test_file)
    item_col = test_file['item_zip']
    buyer_col = test_file['buyer_zip']
    add_zip_distance_column(item_col, buyer_col)

    #add_zip_distance_column('')
    # get_distance('04007', '49036')
    pass
