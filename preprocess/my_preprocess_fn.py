import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import logging

nyc_category_mapping = {
    'Residential': [
        'Home (private)', 'Neighborhood', 'Residential Building (Apartment / Condo)', 
        'Housing Development', 'Hotel'
    ],
    'Commercial/Services': [
        'Food & Drink Shop', 'Burger Joint', 'Coffee Shop', 'Ice Cream Shop',
        'Deli / Bodega', 'Mexican Restaurant', 'American Restaurant', 'BBQ Joint',
        'Fast Food Restaurant', 'Bar', 'Cupcake Shop', 'Candy Store', 'Pizza Place',
        'Sandwich Place', 'German Restaurant', 'Latin American Restaurant', 'Café',
        'Breakfast Spot', 'Malaysian Restaurant', 'Diner', 'Bakery', 'Fried Chicken Joint',
        'Snack Place', 'Seafood Restaurant', 'Salad Place', 'Wings Joint', 'Japanese Restaurant',
        'Falafel Restaurant', 'Middle Eastern Restaurant', 'Asian Restaurant', 'Beer Garden',
        'Ramen /  Noodle House', 'Hot Dog Joint', 'Cajun / Creole Restaurant', 'Mac & Cheese Joint',
        'Korean Restaurant', 'Sushi Restaurant', 'Gastropub', 'Caribbean Restaurant', 
        'African Restaurant', 'Cuban Restaurant', 'Indian Restaurant', 'Dessert Shop',
        'Thai Restaurant', 'Soup Place', 'Taco Place', 'Steakhouse', 'Dumpling Restaurant',
        'Vietnamese Restaurant', 'Southern / Soul Food Restaurant', 'Tapas Restaurant',
        'Filipino Restaurant', 'Brazilian Restaurant', 'Australian Restaurant', 
        'Eastern European Restaurant', 'Swiss Restaurant', 'Dim Sum Restaurant',
        'Mobile Phone Shop', 'Automotive Shop', 'Clothing Store', 'Electronics Store', 
        'Tattoo Parlor', 'Department Store', 'Hardware Store', 'Bookstore', 'Toy / Game Store',
        'Miscellaneous Shop', 'Furniture / Home Store', 'Bridal Shop', 'Paper / Office Supplies Store',
        'Convenience Store', 'Hobby Shop', 'Pet Store', 'Jewelry Store', 'Camera Store', 
        'Thrift / Vintage Store', 'Antique Shop', 'Market', 'Flea Market', 'Garden Center',
        'Salon / Barbershop', 'Cosmetics Shop', 'Bank', 'Financial or Legal Service',
        'Professional & Other Places', 'Design Studio', 'Laundry Service', 'Smoke Shop',
        'Post Office', 'Tattoo Parlor', 'Tanning Salon', 'Government Building', 'Office',
        'Other Nightlife', 'Building', 'Spanish Restaurant', 'Factory', 'Burrito Place',
        'Chinese Restaurant', 'Bagel Shop', 'Vegetarian / Vegan Restaurant', 'Donut Shop',
        'Sporting Goods Shop', 'French Restaurant', 'Italian Restaurant', 'Food Truck', 'Restaurant',
        'Tea Room', 'Brewery', 'Recycling Facility', 'Mediterranean Restaurant', 'Gift Shop', 'Food',
        'South American Restaurant', 'Molecular Gastronomy Restaurant', 'Scandinavian Restaurant',
        'Military Base', 'City'
    ],
    'Educational': [
        'Student Center', 'University', 'College Academic Building', 'Community College',
        'General College & University', 'College & University', 'Library', 'Law School', 
        'Trade School', 'Nursery School', 'Elementary School', 'Middle School', 
        'High School', 'College Stadium', 'School'
    ],
    'Transportation': [
        'Subway', 'Bus Station', 'Light Rail', 'Airport', 'Train Station', 'Parking', 
        'General Travel', 'Rental Car Location', 'Taxi', 'Ferry', 'Road', 'Harbor / Marina',
        'Bridge', 'Gas Station / Garage', 'River', 'Travel', 'Travel & Transport', 'Moving Target'
    ],
    'Culture & Leisure': [
        'Arts & Crafts Store', 'Music Venue', 'Movie Theater', 'Scenic Lookout', 'Theater', 
        'General Entertainment', 'Bowling Alley', 'Arcade', 'Comedy Club', 'Museum', 
        'Performing Arts Venue', 'Event Space', 'Art Museum', 'Concert Hall', 'Zoo', 
        'Aquarium', 'Casino', 'Science Museum', 'Racetrack', 'Fair', 'Music Store',
        'Stadium', 'Art Gallery', 'Park', 'Campground', 'Other Great Outdoors',
        'Beach', 'Playground', 'Pool Hall', 'Plaza', 'Outdoors & Recreation', 
        'Sculpture Garden', 'Garden', 'Travel Lounge', 'Rest Area', 'Convention Center',
        'Historic Site', 'Mall', 'Synagogue', 'Church', 'Cemetery', 'Temple', 'Shrine',
        'Arts & Entertainment', 'Spiritual Center'
    ],
    'Healthcare & Welfare': [
        'Gym / Fitness Center', 'Medical Center', 'Drugstore / Pharmacy', 'Spa / Massage',
        'Athletic & Sport', 'Pool', 'Animal Shelter', 'Funeral Home'
    ]
}

tky_category_mapping = {
    'Residential': [
        'Neighborhood', 'Home (private)', 'Residential Building (Apartment / Condo)',
        'Housing Development', 'Sorority House'
    ],
    'Commercial/Services': [
        'Convention Center', 'Japanese Restaurant', 'Electronics Store', 'Cafï¿½',
        'Fast Food Restaurant', 'Convenience Store', 'Paper / Office Supplies Store',
        'Chinese Restaurant', 'Office', 'Bookstore', 'Hobby Shop', 'Bar',
        'Miscellaneous Shop', 'Toy / Game Store', 'Ramen /  Noodle House', 'Smoke Shop',
        'Shrine', 'Plaza', 'Building', 'Italian Restaurant', 'General Entertainment',
        'Clothing Store', 'Hardware Store', 'Coffee Shop', 'Fried Chicken Joint',
        'Food & Drink Shop', 'Dessert Shop', 'Restaurant', 'Mall', 'Bakery',
        'Indian Restaurant', 'Post Office', 'Government Building',
        'Drugstore / Pharmacy', 'Diner', 'Soup Place', 'Burger Joint', 'Racetrack',
        'Department Store', 'Record Shop', 'Music Venue', 'General Travel',
        'Furniture / Home Store', 'Camera Store', 'Sushi Restaurant', 'Hotel',
        'Arts & Crafts Store', 'Bike Shop', 'Mobile Phone Shop', 'Recycling Facility',
        'Antique Shop', 'Donut Shop', 'Deli / Bodega', 'Ice Cream Shop', 'Asian Restaurant',
        'Steakhouse', 'Video Store', 'Video Game Store', 'Dumpling Restaurant',
        'Sandwich Place', 'Internet Cafe', 'Military Base', 'Sporting Goods Shop',
        'Bank', 'Music Store', 'Travel Lounge', 'Seafood Restaurant',
        'Travel & Transport', 'Breakfast Spot', 'Gift Shop', 'Athletic & Sport',
        'Pizza Place', 'BBQ Joint', 'Gaming Cafe', 'Salon / Barbershop', 'Hot Dog Joint',
        'American Restaurant', 'Brewery', 'Harbor / Marina', 'Middle Eastern Restaurant',
        'Automotive Shop', 'Fish & Chips Shop', 'Comedy Club', 'Gastropub',
        'Scenic Lookout', 'Caribbean Restaurant', 'Shop & Service', 'French Restaurant',
        'Thai Restaurant', 'Brazilian Restaurant', 'Moving Target', 'Laundry Service',
        'Flower Shop', 'River', 'Spiritual Center', 'Playground', 'Mexican Restaurant',
        'Car Dealership', 'Candy Store', 'Food', 'Motorcycle Shop', 'Wings Joint',
        'Tea Room', 'Board Shop', 'Mediterranean Restaurant', 'Tanning Salon',
        'Food Truck', 'Thrift / Vintage Store', 'Pool', 'Embassy / Consulate',
        'Snack Place', 'Professional & Other Places', 'Korean Restaurant',
        'Cosmetics Shop', 'Factory', 'Pet Store', 'Bike Rental / Bike Share'
    ],
    'Educational': [
        'University', 'College Academic Building', 'General College & University',
        'Student Center', 'School', 'High School', 'Community College', 'Trade School',
        'Elementary School', 'Medical School', 'College & University', 'Nursery School',
        'College Stadium'
    ],
    'Transportation': [
        'Train Station', 'Subway', 'Bus Station', 'Road', 'Light Rail',
        'Gas Station / Garage', 'Rest Area', 'Parking', 'Airport', 'Ferry', 'Taxi',
        'Bridge'
    ],
    'Culture & Leisure': [
        'Event Space', 'Stadium', 'Arcade', 'Temple', 'Park',
        'Other Great Outdoors', 'Spa / Massage', 'Movie Theater', 'Sculpture Garden',
        'Aquarium', 'Zoo', 'Art Museum', 'Performing Arts Venue', 'Library',
        'Science Museum', 'Church', 'Historic Site', 'History Museum', 'Bowling Alley',
        'Garden', 'Concert Hall', 'Casino', 'Other Nightlife', 'Art Gallery',
        'Beer Garden', 'Theater', 'Museum', 'Beach', 'Public Art', 'Garden Center',
        'Outdoors & Recreation', 'Nightlife Spot', 'Cemetery'
    ],
    'Healthcare & Welfare': [
        'Medical Center', 'Gym / Fitness Center'
    ]
}

def add_holidays_nyc(df):
    """Add holiday information to NYC data.

    Args:
        df (pd.DataFrame): Input dataframe containing NYC data.

    Returns:
        _type_: _description_
    """
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])

    # 미국 공휴일 정의 (2012년 4월~2013년 2월)
    us_holidays = [
        datetime(2012, 4, 6).date(),  # Good Friday
        datetime(2012, 5, 28).date(), # Memorial Day
        datetime(2012, 7, 4).date(),  # Independence Day
        datetime(2012, 9, 3).date(),  # Labor Day
        datetime(2012, 10, 8).date(), # Columbus Day
        datetime(2012, 11, 12).date(),# Veterans Day (대체휴일)
        datetime(2012, 11, 22).date(),# Thanksgiving
        datetime(2012, 12, 25).date(),# Christmas
        datetime(2013, 1, 1).date(),  # New Year's Day
        datetime(2013, 1, 21).date(), # Martin Luther King Jr. Day
        datetime(2013, 2, 18).date()  # Presidents’ Day
    ]
    
    # 공휴일 정보 추가
    df['is_weekend'] = df['LocalTime'].dt.weekday.isin([5, 6])  # 토요일(5), 일요일(6)
    df['is_holiday'] = df['LocalTime'].dt.date.isin(us_holidays)

    df['Holiday'] = df['is_weekend'] | df['is_holiday']
    df.drop(columns=['is_weekend', 'is_holiday'], inplace=True)
    
    return df

# 일본 공휴일 정보 추가
def add_holidays_tky(df):
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])  # 또는 다른 시간 컬럼
    
    japan_holidays = [
        # 2012
        datetime(2012, 4, 29).date(), datetime(2012, 4, 30).date(),
        datetime(2012, 5, 3).date(), datetime(2012, 5, 4).date(), datetime(2012, 5, 5).date(),
        datetime(2012, 7, 16).date(), datetime(2012, 9, 17).date(), datetime(2012, 9, 22).date(),
        datetime(2012, 10, 8).date(), datetime(2012, 11, 3).date(), datetime(2012, 11, 23).date(),
        datetime(2012, 12, 23).date(), datetime(2012, 12, 24).date(),
        # 2013
        datetime(2013, 1, 1).date(), datetime(2013, 1, 14).date(), datetime(2013, 2, 11).date()
    ]
    
    df['date_only'] = df['LocalTime'].dt.date
    df['is_weekend'] = df['LocalTime'].dt.weekday.isin([5, 6])  # 토/일
    df['is_holiday'] = df['date_only'].isin(japan_holidays)
    
    # 주말 or 공휴일
    df['Holiday'] = df['is_weekend'] | df['is_holiday']
    
    # 하루 기준 상대 시간 추가
    df['NormInDayTime'] = (df['LocalTime'].dt.hour * 3600 + df['LocalTime'].dt.minute * 60 + df['LocalTime'].dt.second) / (24*3600)  # 하루를 1로 정규화
    
    # 정리
    df.drop(columns=['is_weekend', 'is_holiday', 'date_only'], inplace=True)
    
    return df


# 카테고리 매핑 함수
def map_to_upper_category(category_name, category_mapping):
    for key, values in category_mapping.items():
        if category_name in values:
            return key
    return 'Else'

# 상위 카테고리 맵핑
def make_uppercat(df, opt):
    if opt == 'NYC':
        category_mapping = nyc_category_mapping
    elif opt == 'TKY':
        category_mapping = tky_category_mapping
        
    df['UpperCategory'] = df['PoiCategoryName'].apply(map_to_upper_category, args=(category_mapping,))

    return df


def id_encode(
        fit_df: pd.DataFrame,
        encode_df: pd.DataFrame,
        column: str,
        padding: int = -1
) -> Tuple[LabelEncoder, int]:
    """

    :param fit_df: only consider the data in encode df for constructing LabelEncoder instance
    :param encode_df: the dataframe which use the constructed LabelEncoder instance to encode their values
    :param column: the column to be encoded
    :param padding:
    :return:
    """
    id_le = LabelEncoder()
    id_le = id_le.fit(fit_df[column].values.tolist())
    if padding == 0:
        padding_id = padding
        encode_df[column] = [
            id_le.transform([i])[0] + 1 if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    else:
        padding_id = len(id_le.classes_)
        encode_df[column] = [
            id_le.transform([i])[0] if i in id_le.classes_ else padding_id
            for i in encode_df[column].values.tolist()
        ]
    return id_le, padding_id


def ignore_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ignore the first check-in sample of every trajectory because of no historical check-in.

    """
    df['pseudo_session_trajectory_rank'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].rank(method='first')
    df['query_pseudo_session_trajectory_id'] = df['pseudo_session_trajectory_id'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'query_pseudo_session_trajectory_id'] = None
    df['last_checkin_epoch_time'] = df['UTCTimeOffsetEpoch'].shift()
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'last_checkin_epoch_time'] = None
    df.loc[df['UserRank'] == 1, 'SplitTag'] = 'ignore'
    df.loc[df['pseudo_session_trajectory_rank'] == 1, 'SplitTag'] = 'ignore'
    return df


def only_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only keep the last check-in samples in validation and testing for measuring model performance.

    """
    df['pseudo_session_trajectory_count'] = df.groupby(
        'pseudo_session_trajectory_id')['UTCTimeOffset'].transform('count')
    df.loc[(df['SplitTag'] == 'validation') & (
            df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
    ), 'SplitTag'] = 'ignore'
    df.loc[(df['SplitTag'] == 'test') & (
            df['pseudo_session_trajectory_count'] != df['pseudo_session_trajectory_rank']
    ), 'SplitTag'] = 'ignore'
    return df


def remove_unseen_user_poi(df: pd.DataFrame) -> Dict:
    """
    Remove the samples of Validate and Test if those POIs or Users didnt show in training samples

    """
    preprocess_result = dict()
    df_train = df[df['SplitTag'] == 'train']
    df_validate = df[df['SplitTag'] == 'validation']
    df_test = df[df['SplitTag'] == 'test']

    train_user_set = set(df_train['UserId'])
    train_poi_set = set(df_train['PoiId'])
    df_validate = df_validate[
        (df_validate['UserId'].isin(train_user_set)) & (df_validate['PoiId'].isin(train_poi_set))].reset_index()
    df_test = df_test[(df_test['UserId'].isin(train_user_set)) & (df_test['PoiId'].isin(train_poi_set))].reset_index()

    preprocess_result['sample'] = df
    preprocess_result['train_sample'] = df_train
    preprocess_result['validate_sample'] = df_validate
    preprocess_result['test_sample'] = df_test

    logging.info(
        f"[Preprocess] train shape: {df_train.shape}, validation shape: {df_validate.shape}, "
        f"test shape: {df_test.shape}"
    )
    return preprocess_result
