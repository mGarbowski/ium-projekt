from typing import Union, Optional

from pydantic import BaseModel, Field

Number = Union[int, float]


class Listing(BaseModel):
    id: str
    listing_url: str
    source: str
    name: str
    description: Optional[str]
    neighborhood_overview: Optional[str]
    picture_url: str
    host_id: str
    host_url: str
    host_name: Optional[str]
    host_since: Optional[str]
    host_location: Optional[str]
    host_about: Optional[str]
    host_response_time: Optional[str]
    host_response_rate: Optional[str]
    host_acceptance_rate: Optional[str]
    host_is_superhost: Optional[str]
    host_thumbnail_url: Optional[str]
    host_picture_url: Optional[str]
    host_neighbourhood: Optional[str]
    host_listings_count: Optional[Number]
    host_total_listings_count: Optional[Number]
    host_verifications: Optional[str]
    host_has_profile_pic: Optional[str]
    host_identity_verified: Optional[str]
    neighbourhood: Optional[str]
    neighbourhood_cleansed: str
    neighbourhood_group_cleansed: str
    latitude: float
    longitude: float
    property_type: str
    room_type: str
    accommodates: Number
    bathrooms: Optional[Number]
    bathrooms_text: Optional[str]
    bedrooms: Optional[Number]
    beds: Optional[Number]
    amenities: str
    price: Optional[Union[str, Number]]
    minimum_nights: Number
    maximum_nights: Number
    minimum_minimum_nights: Number
    maximum_minimum_nights: Number
    minimum_maximum_nights: Number
    maximum_maximum_nights: Number
    minimum_nights_avg_ntm: Number
    maximum_nights_avg_ntm: Number
    has_availability: Optional[str]
    availability_30: Number
    availability_60: Number
    availability_90: Number
    availability_365: Number
    license: Optional[str]
    instant_bookable: str
    calculated_host_listings_count: Number
    calculated_host_listings_count_entire_homes: Number
    calculated_host_listings_count_private_rooms: Number
    calculated_host_listings_count_shared_rooms: Number


class PredictRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    data: Listing = Field(..., description="Features for prediction")
