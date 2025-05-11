from typing import Union

from pydantic import BaseModel, Field

Number = Union[int, float]


class Listing(BaseModel):  # TODO remove or make optional unused fields
    id: str
    listing_url: str
    scrape_id: str
    last_scraped: str
    source: str
    name: str
    description: str
    neighborhood_overview: str
    picture_url: str
    host_id: str
    host_url: str
    host_name: str
    host_since: str
    host_location: str
    host_about: str
    host_response_time: str
    host_response_rate: str
    host_acceptance_rate: str
    host_is_superhost: str
    host_thumbnail_url: str
    host_picture_url: str
    host_neighbourhood: str
    host_listings_count: Number
    host_total_listings_count: Number
    host_verifications: str
    host_has_profile_pic: str
    host_identity_verified: str
    neighbourhood: str
    neighbourhood_cleansed: str
    neighbourhood_group_cleansed: str
    latitude: str
    longitude: str
    property_type: str
    room_type: str
    accommodates: Number
    bathrooms: Number
    bathrooms_text: str
    bedrooms: Number
    beds: Number
    amenities: str
    price: Union[str, Number]
    minimum_nights: Number
    maximum_nights: Number
    minimum_minimum_nights: Number
    maximum_minimum_nights: Number
    minimum_maximum_nights: Number
    maximum_maximum_nights: Number
    minimum_nights_avg_ntm: Number
    maximum_nights_avg_ntm: Number
    has_availability: str
    availability_30: Number
    availability_60: Number
    availability_90: Number
    availability_365: Number
    license: str
    instant_bookable: str
    calculated_host_listings_count: Number
    calculated_host_listings_count_entire_homes: Number
    calculated_host_listings_count_private_rooms: Number
    calculated_host_listings_count_shared_rooms: Number


class PredictRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    data: Listing = Field(..., description="Features for prediction")
