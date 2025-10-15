"""
Region Selection API for City-based Biomass Analysis
Handles city subdivision into regions and satellite preview generation
"""

import logging
from typing import Dict, List, Tuple, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

# Import existing modules
from app.models.gee_data_fetcher import GEEDataFetcher
from app.api.satellite_image_generator import fetch_high_res_satellite_and_ndvi

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize data fetcher
gee_fetcher = GEEDataFetcher()

class RegionRequest(BaseModel):
    """Model for region generation request"""
    city: str

class RegionInfo(BaseModel):
    """Model for region information"""
    id: str
    name: str
    description: str
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    coordinates: Dict
    preview_image_url: Optional[str] = None

class RegionResponse(BaseModel):
    """Model for region selection response"""
    city: str
    total_regions: int
    regions: List[RegionInfo]
    city_center: List[float]  # [lat, lon]
    city_bbox: List[float]

def get_city_specific_descriptions(city_name: str) -> Dict[str, str]:
    """
    Get landmark-specific descriptions for each region of major Indian cities
    """
    city_lower = city_name.lower().strip()
    
    descriptions = {
        # Chennai
        'chennai': {
            'center': 'Central business district featuring T. Nagar shopping hub, Nungambakkam commercial area, Egmore railway station, and Anna Salai (Mount Road) — includes Express Avenue Mall, Spencer Plaza, and government offices.',
            'north': 'Northern region covering Chennai Port, George Town wholesale markets, Washermenpet residential areas, and Royapuram — includes Chennai Central Railway Station, High Court, and traditional trading districts.',
            'south': 'Southern zone including Guindy National Park, Velachery IT corridor, Tambaram railway junction, and Saidapet — features IIT Madras, St. Thomas Mount, and OMR tech parks.',
            'east': 'Eastern coastal area with Marina Beach, Mylapore Kapaleeshwarar Temple, Adyar Theosophical Society, and Besant Nagar — includes Elliot\'s Beach, Santhome Cathedral, and cultural centers.',
            'west': 'Western suburbs covering Anna Nagar residential area, Koyambedu wholesale market, Ambattur industrial estate, and Mogappair — includes Chennai Metro hub and expanding residential projects.'
        },
        
        # Mumbai
        'mumbai': {
            'center': 'Financial district including Nariman Point business hub, Fort heritage area, Churchgate station, and CST terminus — features Bombay Stock Exchange, Reserve Bank, and colonial architecture.',
            'north': 'Northern suburbs with Bandra film city, Andheri international airport, Borivali National Park, and Malad residential areas — includes Bollywood studios, shopping malls, and upscale housing.',
            'south': 'Southern peninsula featuring Colaba Gateway of India, Cuffe Parade luxury towers, Malabar Hill residences, and Worli Sea Link — includes Taj Hotel, Marine Drive, and affluent neighborhoods.',
            'east': 'Eastern corridor covering Kurla business district, Ghatkopar metro hub, Vikhroli industrial area, and Powai IT parks — includes manufacturing units, residential complexes, and tech companies.',
            'west': 'Western coastline from Worli BKC financial district, Bandra linking road, Juhu beach area, to Versova fishing village — includes film studios, beaches, and premium residential towers.'
        },
        
        # Delhi
        'delhi': {
            'center': 'Central Delhi with Connaught Place shopping circle, India Gate memorial, Lutyens\' Delhi government area, and Rajpath ceremonial avenue — includes Parliament House, Rashtrapati Bhavan, and administrative buildings.',
            'north': 'North Delhi covering Red Fort historic monument, Chandni Chowk traditional market, Delhi University campus, and Civil Lines residential area — includes Jama Masjid, Kashmere Gate, and old city bazaars.',
            'south': 'South Delhi featuring Hauz Khas village, Greater Kailash markets, Vasant Kunj residential area, and Nehru Place IT hub — includes Select City Walk, DLF malls, and upscale neighborhoods.',
            'east': 'East Delhi including Laxmi Nagar market area, Preet Vihar residential colony, Mayur Vihar housing complex, and Akshardham temple — includes metro stations, schools, and dense residential areas.',
            'west': 'West Delhi covering Rajouri Garden market, Janakpuri residential area, Dwarka sub-city, and Indira Gandhi Airport — includes metro hub, shopping centers, and modern planned developments.'
        },
        
        # Bangalore/Bengaluru
        'bangalore': {
            'center': 'Central Bangalore with MG Road shopping street, Brigade Road commercial area, Cubbon Park green space, and Vidhana Soudha government building — includes UB City Mall, pubs, and business districts.',
            'north': 'North Bangalore covering Hebbal flyover junction, Yelahanka residential area, Devanahalli international airport, and Bagalur IT parks — includes emerging residential projects and tech campuses.',
            'south': 'South Bangalore featuring Koramangala startup hub, BTM Layout residential area, Bannerghatta National Park, and JP Nagar — includes tech companies, cafes, and young professional neighborhoods.',
            'east': 'East Bangalore including Whitefield IT corridor, Marathahalli tech hub, KR Puram railway station, and ITPL tech park — includes international companies, residential complexes, and shopping malls.',
            'west': 'West Bangalore covering Rajajinagar residential area, Vijayanagar traditional neighborhood, Peenya industrial area, and Yeshwantpur railway junction — includes manufacturing units and local markets.'
        },
        
        # Hyderabad
        'hyderabad': {
            'center': 'Central Hyderabad with Charminar historic monument, Abids shopping area, Nampally railway station, and Secunderabad cantonment — includes traditional bazaars, government offices, and Nizami heritage sites.',
            'north': 'Northern region covering Begumpet airport area, Punjagutta commercial district, Kukatpally residential hub, and Ameerpet educational zone — includes shopping malls, corporate offices, and coaching centers.',
            'south': 'Southern corridor featuring Gachibowli HITEC City, Madhapur Financial District, Kondapur IT hub, and Jubilee Hills residential area — includes global tech companies, modern infrastructure, and upscale housing.',
            'east': 'Eastern area including Uppal residential area, LB Nagar commercial zone, Dilsukhnagar market district, and Ramanthapur — includes metro connectivity, local markets, and middle-class neighborhoods.',
            'west': 'Western zone covering Mehdipatnam junction, Tolichowki residential area, Manikonda IT corridor, and Shaikpet — includes educational institutions, growing residential projects, and commercial developments.'
        },
        
        # Kolkata
        'kolkata': {
            'center': 'Central Kolkata with Park Street restaurant hub, Esplanade commercial area, Dalhousie business district, and BBD Bagh government zone — includes Victoria Memorial, colonial buildings, and cultural centers.',
            'north': 'North Kolkata covering Shyambazar traditional area, Bagbazar heritage neighborhood, Sovabazar cultural district, and College Street book market — includes heritage buildings, local markets, and educational institutions.',
            'south': 'South Kolkata featuring Ballygunge residential area, Gariahat shopping district, Jadavpur University campus, and Tollygunge film city — includes shopping centers, restaurants, and upscale neighborhoods.',
            'east': 'East Kolkata including Salt Lake planned city, Rajarhat IT hub, New Town modern development, and Bidhannagar — includes IT sector, residential complexes, and contemporary infrastructure.',
            'west': 'West Kolkata covering Behala residential area, Thakurpukur suburban zone, Joka educational hub, and Garia — includes expanding suburban developments, local markets, and growing connectivity.'
        },
        
        # Pune
        'pune': {
            'center': 'Central Pune with Shivajinagar commercial hub, Camp cantonment area, Deccan Gymkhana educational zone, and JM Road shopping street — includes Shaniwar Wada palace, FC Road, and traditional markets.',
            'north': 'North Pune covering Aundh residential area, Baner IT corridor, Wakad tech hub, and Pimpri-Chinchwad industrial zone — includes tech parks, modern housing projects, and commercial developments.',
            'south': 'South Pune featuring Kothrud residential area, Warje housing projects, Sinhagad Road IT corridor, and Katraj — includes educational institutions, residential developments, and proximity to hill stations.',
            'east': 'East Pune including Kharadi IT hub, Wagholi residential township, Lohegaon airport area, and Viman Nagar — includes tech companies, new residential projects, and international airport.',
            'west': 'West Pune covering Hinjewadi IT destination, Balewadi sports complex, Sus residential area, and Rajiv Gandhi Infotech Park — includes major IT companies, modern infrastructure, and planned communities.'
        },
        
        # Ahmedabad
        'ahmedabad': {
            'center': 'Central Ahmedabad with Lal Darwaja heritage area, Manek Chowk traditional market, Ellis Bridge commercial zone, and Teen Darwaza historic gate — includes traditional pols, textile markets, and Gujarati heritage sites.',
            'north': 'North Ahmedabad covering Sabarmati Ashram memorial, Chandkheda residential area, Motera cricket stadium, and Kalol industrial zone — includes developing infrastructure and residential projects.',
            'south': 'South Ahmedabad featuring Navrangpura commercial area, Vastrapur residential zone, Bodakdev upscale neighborhood, and Law Garden market — includes shopping malls, restaurants, and modern residential complexes.',
            'east': 'East Ahmedabad including Maninagar residential area, Nikol industrial zone, Vastral textile hub, and Odhav manufacturing district — includes textile mills, manufacturing units, and working-class neighborhoods.',
            'west': 'West Ahmedabad covering Bopal residential area, Shela luxury housing, SG Highway IT corridor, and Prahlad Nagar commercial zone — includes tech parks, luxury projects, and modern commercial developments.'
        },
        
        # Jaipur
        'jaipur': {
            'center': 'Central Jaipur with City Palace royal complex, Hawa Mahal wind palace, Pink City UNESCO heritage area, and Johari Bazaar traditional market — includes royal palaces, traditional bazaars, and Rajasthani architecture.',
            'north': 'North Jaipur covering Civil Lines administrative area, Jyoti Nagar residential colony, Shastri Nagar housing area, and Adarsh Nagar — includes government offices, residential colonies, and educational institutions.',
            'south': 'South Jaipur featuring Malviya Nagar residential area, Jagatpura housing development, Mansarovar planned colony, and World Trade Park — includes shopping centers, hospitals, and contemporary infrastructure.',
            'east': 'East Jaipur including Vidhyadhar Nagar planned area, Lalkothi residential zone, Nirman Nagar housing colony, and Murlipura — includes well-organized neighborhoods, parks, and schools.',
            'west': 'West Jaipur covering Vaishali Nagar residential area, Ajmer Road commercial corridor, Sodala industrial zone, and transportation hubs — includes markets, industrial areas, and connectivity points.'
        },
        
        # Surat
        'surat': {
            'center': 'Central Surat with Chowk Bazaar diamond market, Salabatpura textile hub, Ring Road commercial area, and traditional business centers — includes diamond cutting units, textile markets, and Gujarati commercial districts.',
            'north': 'North Surat covering Adajan residential area, Pal upscale neighborhood, Althan luxury housing, and modern amenities — includes shopping complexes, well-planned infrastructure, and contemporary developments.',
            'south': 'South Surat featuring Udhna industrial area, Sachin manufacturing zone, Pandesara chemical hub, and textile mills — includes industrial corridors, manufacturing units, and economic growth centers.',
            'east': 'East Surat including Katargam residential area, Varachha dense neighborhood, Kapodra traditional community, and local markets — includes residential areas, small businesses, and traditional neighborhood communities.',
            'west': 'West Surat covering Vesu upscale area, Dumas beach resort, Magdalla coastal zone, and luxury residential projects — includes beaches, luxury housing, and emerging commercial developments.'
        },
        
        # Gandhinagar
        'gandhinagar': {
            'center': 'Central Gandhinagar with Secretariat complex, Assembly building, Mahatma Mandir convention center, and Akshardham temple — includes government offices, planned sectors, and administrative buildings.',
            'north': 'Northern Gandhinagar covering Sector 1-10 residential areas, Indroda Nature Park, Children\'s University, and GIFT City approach — includes educational institutions, parks, and planned developments.',
            'south': 'Southern Gandhinagar featuring Sector 15-30 residential zones, industrial areas, Sardar Vallabhbhai Patel Stadium, and commercial districts — includes housing developments and sports facilities.',
            'east': 'Eastern Gandhinagar including Pethapur industrial area, Kalol manufacturing zone, residential sectors, and transportation corridors — includes industrial developments and connectivity routes.',
            'west': 'Western Gandhinagar covering GIFT City financial district, Sabarmati riverfront, planned residential sectors, and commercial zones — includes modern infrastructure and financial hub.'
        },
        
        # Patna
        'patna': {
            'center': 'Central Patna with Gandhi Maidan historic ground, Patna Junction railway station, Boring Road commercial area, and Secretariat complex — includes government offices, traditional markets, and administrative buildings.',
            'north': 'Northern Patna covering Kankarbagh residential area, Patna University campus, Rajendra Nagar colony, and educational institutions — includes colleges, residential developments, and academic centers.',
            'south': 'Southern Patna featuring Danapur cantonment area, industrial zones, manufacturing districts, and residential colonies — includes military establishments, industrial areas, and housing developments.',
            'east': 'Eastern Patna including Kurji residential area, Patna City traditional zone, local markets, and dense neighborhoods — includes old city areas, traditional businesses, and local communities.',
            'west': 'Western Patna covering Bailey Road commercial corridor, Boring Canal Road, residential areas, and developing infrastructure — includes shopping areas, residential colonies, and modern developments.'
        },
        
        # Lucknow
        'lucknow': {
            'center': 'Central Lucknow with Hazratganj shopping area, Charbagh railway station, Vidhan Sabha building, and Bara Imambara — includes government offices, traditional markets, and Nawabi heritage sites.',
            'north': 'Northern Lucknow covering Alambagh commercial hub, Aliganj residential area, Kalyanpur industrial zone, and educational institutions — includes shopping centers, residential developments, and colleges.',
            'south': 'Southern Lucknow featuring Gomti Nagar planned city, IT parks, modern residential areas, and commercial complexes — includes tech companies, upscale housing, and contemporary infrastructure.',
            'east': 'Eastern Lucknow including Aminabad traditional market, old city areas, Chowk heritage zone, and dense residential neighborhoods — includes traditional bazaars, heritage buildings, and local communities.',
            'west': 'Western Lucknow covering Mahanagar residential area, Indira Nagar colony, commercial zones, and developing infrastructure — includes residential projects, markets, and connectivity routes.'
        },
        
        # Bhopal
        'bhopal': {
            'center': 'Central Bhopal with New Market shopping area, Bhopal Junction railway station, Vallabh Bhawan secretariat, and Upper Lake — includes government offices, commercial centers, and historic lakes.',
            'north': 'Northern Bhopal covering Kolar residential area, industrial zones, manufacturing districts, and educational institutions — includes colleges, industrial developments, and residential colonies.',
            'south': 'Southern Bhopal featuring Arera Colony upscale area, IT parks, modern residential developments, and commercial complexes — includes tech companies, shopping malls, and contemporary housing.',
            'east': 'Eastern Bhopal including Habibganj railway station, residential areas, local markets, and developing zones — includes transportation hubs, housing developments, and commercial areas.',
            'west': 'Western Bhopal covering Bairagarh residential area, industrial zones, educational institutions, and connectivity routes — includes residential projects, colleges, and infrastructure developments.'
        },
        
        # Thiruvananthapuram
        'thiruvananthapuram': {
            'center': 'Central Thiruvananthapuram with Padmanabhaswamy Temple, Secretariat complex, Central Station, and MG Road — includes government offices, heritage sites, and commercial areas.',
            'north': 'Northern Thiruvananthapuram covering Pattom IT hub, medical college area, residential zones, and educational institutions — includes tech parks, hospitals, colleges, and housing developments.',
            'south': 'Southern Thiruvananthapuram featuring Kovalam beach resort, Vizhinjam port area, coastal regions, and tourism zones — includes beaches, fishing villages, and hospitality infrastructure.',
            'east': 'Eastern Thiruvananthapuram including Karamana residential area, traditional neighborhoods, local markets, and cultural centers — includes residential colonies, local businesses, and community areas.',
            'west': 'Western Thiruvananthapuram covering airport area, coastal regions, Shanghumukham beach, and transportation hubs — includes international airport, beaches, and connectivity infrastructure.'
        },
        
        # Bhubaneswar
        'bhubaneswar': {
            'center': 'Central Bhubaneswar with Lingaraj Temple complex, Master Canteen area, Rajmahal Square, and government offices — includes ancient temples, administrative buildings, and traditional markets.',
            'north': 'Northern Bhubaneswar covering Nayapalli residential area, Jaydev Vihar educational zone, modern housing developments, and commercial complexes — includes colleges, shopping centers, and contemporary infrastructure.',
            'south': 'Southern Bhubaneswar featuring Infocity IT hub, Chandrasekharpur residential area, tech parks, and modern developments — includes IT companies, upscale housing, and business districts.',
            'east': 'Eastern Bhubaneswar including Old Town temple area, traditional neighborhoods, Bindusagar lake, and heritage zones — includes ancient temples, cultural sites, and traditional communities.',
            'west': 'Western Bhubaneswar covering Saheed Nagar residential area, Unit areas planned development, commercial zones, and connectivity routes — includes residential projects, markets, and infrastructure.'
        },
        
        # Ranchi
        'ranchi': {
            'center': 'Central Ranchi with Main Road commercial area, Ranchi Junction railway station, Secretariat complex, and traditional markets — includes government offices, business centers, and administrative buildings.',
            'north': 'Northern Ranchi covering Hinoo residential area, industrial zones, educational institutions, and developing areas — includes colleges, residential developments, and commercial projects.',
            'south': 'Southern Ranchi featuring Morabadi sports complex, residential colonies, educational institutions, and recreational areas — includes sports facilities, housing developments, and schools.',
            'east': 'Eastern Ranchi including Lalpur residential area, local markets, traditional neighborhoods, and community centers — includes residential areas, local businesses, and cultural sites.',
            'west': 'Western Ranchi covering Kanke residential area, industrial zones, educational institutions, and connectivity routes — includes colleges, industrial developments, and transportation infrastructure.'
        },
        
        # Raipur
        'raipur': {
            'center': 'Central Raipur with Pandri commercial area, Raipur Junction railway station, Civil Lines administrative zone, and traditional markets — includes government offices, business centers, and commercial districts.',
            'north': 'Northern Raipur covering Shankar Nagar residential area, educational institutions, industrial zones, and developing areas — includes colleges, housing developments, and commercial projects.',
            'south': 'Southern Raipur featuring Telibandha residential area, IT parks, modern developments, and commercial complexes — includes tech companies, contemporary housing, and business districts.',
            'east': 'Eastern Raipur including Moudhapara residential area, local markets, traditional neighborhoods, and community centers — includes residential colonies, local businesses, and cultural areas.',
            'west': 'Western Raipur covering Devendra Nagar residential area, industrial zones, educational institutions, and connectivity infrastructure — includes residential projects, colleges, and transportation routes.'
        },
        
        # Panaji
        'panaji': {
            'center': 'Central Panaji with Church Square heritage area, Secretariat complex, Mandovi riverfront, and commercial districts — includes government offices, heritage buildings, and business centers.',
            'north': 'Northern Panaji covering Altinho residential area, educational institutions, government quarters, and developing zones — includes colleges, housing developments, and administrative areas.',
            'south': 'Southern Panaji featuring Miramar beach area, residential zones, hospitality infrastructure, and coastal regions — includes beaches, hotels, and tourism facilities.',
            'east': 'Eastern Panaji including Ribandar residential area, industrial zones, local markets, and traditional neighborhoods — includes residential colonies, local businesses, and community areas.',
            'west': 'Western Panaji covering Dona Paula tourist area, educational institutions, coastal regions, and connectivity routes — includes tourist attractions, colleges, and transportation infrastructure.'
        },
        
        # Shimla
        'shimla': {
            'center': 'Central Shimla with Mall Road shopping area, Ridge open space, Christ Church, and Secretariat complex — includes government offices, heritage buildings, and commercial centers.',
            'north': 'Northern Shimla covering Sanjauli residential area, educational institutions, developing zones, and hill slopes — includes colleges, housing developments, and scenic areas.',
            'south': 'Southern Shimla featuring Lakkar Bazaar traditional market, residential areas, local businesses, and cultural centers — includes traditional markets, housing colonies, and community areas.',
            'east': 'Eastern Shimla including Totu residential area, educational institutions, local markets, and traditional neighborhoods — includes colleges, residential developments, and local businesses.',
            'west': 'Western Shimla covering Boileauganj residential area, industrial zones, connectivity routes, and developing infrastructure — includes residential projects, transportation hubs, and modern developments.'
        },
        
        # Vadodara
        'vadodara': {
            'center': 'Central Vadodara with Sayajigunj commercial area, Laxmi Vilas Palace, MS University campus, and Mandvi area — includes heritage buildings, educational institutions, and traditional markets.',
            'north': 'Northern Vadodara covering Alkapuri residential area, Fatehgunj commercial zone, Productivity Road, and modern developments — includes shopping centers, residential complexes, and business districts.',
            'south': 'Southern Vadodara featuring Makarpura industrial area, GIDC estates, Waghodia Road corridor, and manufacturing zones — includes industrial parks, chemical plants, and residential townships.',
            'east': 'Eastern Vadodara including Karelibaug residential area, Gotri area, VIP Road corridor, and educational institutions — includes housing developments, colleges, and commercial centers.',
            'west': 'Western Vadodara covering Nizampura residential area, Harni area, Sama-Savli Road, and developing zones — includes residential projects, local markets, and connectivity infrastructure.'
        },
        
        # Indore
        'indore': {
            'center': 'Central Indore with Rajwada palace, Sarafa Bazaar, Khajrana Ganesh Temple, and MG Road commercial area — includes heritage sites, traditional markets, and business centers.',
            'north': 'Northern Indore covering Vijay Nagar residential area, AB Road commercial corridor, Bhawar Kuan square, and modern developments — includes shopping malls, residential complexes, and IT parks.',
            'south': 'Southern Indore featuring Rau area, Dewas Road corridor, industrial zones, and expanding residential areas — includes manufacturing units, housing projects, and commercial developments.',
            'east': 'Eastern Indore including Palasia area, Ring Road corridor, educational institutions, and residential colonies — includes colleges, housing developments, and local markets.',
            'west': 'Western Indore covering Scheme 78 area, Nipania residential zone, Super Corridor, and IT hubs — includes modern residential projects, tech parks, and commercial complexes.'
        },
        
        # Nagpur
        'nagpur': {
            'center': 'Central Nagpur with Sitabuldi commercial area, Zero Mile marker, Deekshabhoomi Buddhist monument, and Civil Lines — includes government offices, heritage sites, and business districts.',
            'north': 'Northern Nagpur covering Dharampeth residential area, Ramdaspeth colony, medical college zone, and educational institutions — includes hospitals, colleges, and upscale residential areas.',
            'south': 'Southern Nagpur featuring Hingna industrial area, MIDC estates, Wardha Road corridor, and manufacturing zones — includes industrial parks, logistics hubs, and residential townships.',
            'east': 'Eastern Nagpur including Manish Nagar area, Koradi Road, thermal power station zone, and developing areas — includes power infrastructure, residential projects, and commercial centers.',
            'west': 'Western Nagpur covering Bajaj Nagar area, Amravati Road corridor, Wadi area, and residential developments — includes housing colonies, local markets, and connectivity routes.'
        },
        
        # Kanpur
        'kanpur': {
            'center': 'Central Kanpur with Mall Road commercial area, Kanpur Central railway station, Phool Bagh area, and traditional markets — includes government offices, business centers, and heritage sites.',
            'north': 'Northern Kanpur covering Civil Lines area, Cantonment zone, IIT Kanpur campus, and upscale residential areas — includes educational institutions, military establishments, and modern developments.',
            'south': 'Southern Kanpur featuring industrial areas, leather manufacturing zones, Panki area, and residential colonies — includes tanneries, manufacturing units, and working-class neighborhoods.',
            'east': 'Eastern Kanpur including Kalyanpur area, GT Road corridor, educational institutions, and developing zones — includes colleges, residential projects, and commercial areas.',
            'west': 'Western Kanpur covering Kidwai Nagar area, Swaroop Nagar colony, industrial zones, and connectivity infrastructure — includes residential developments, local markets, and transportation hubs.'
        },
        
        # Thane
        'thane': {
            'center': 'Central Thane with Naupada area, Thane railway station, Kopineshwar Temple, and commercial districts — includes government offices, traditional markets, and business centers.',
            'north': 'Northern Thane covering Ghodbunder Road corridor, Kasarvadavali area, IT parks, and modern residential projects — includes tech companies, shopping malls, and contemporary housing.',
            'south': 'Southern Thane featuring Mulund area, LBS Road corridor, residential colonies, and commercial developments — includes housing projects, local markets, and connectivity routes.',
            'east': 'Eastern Thane including Dombivli area, Kalyan connection, industrial zones, and expanding residential areas — includes manufacturing units, housing developments, and transportation hubs.',
            'west': 'Western Thane covering Hiranandani Estate, Ghodbunder area, creek-side developments, and upscale residential projects — includes luxury housing, commercial complexes, and modern infrastructure.'
        },
        
        # Ludhiana
        'ludhiana': {
            'center': 'Central Ludhiana with Chaura Bazaar market, Clock Tower area, Ludhiana Junction railway station, and commercial districts — includes traditional markets, business centers, and government offices.',
            'north': 'Northern Ludhiana covering Civil Lines area, Model Town extension, PAU campus, and residential developments — includes agricultural university, upscale housing, and educational institutions.',
            'south': 'Southern Ludhiana featuring industrial areas, Focal Point manufacturing zone, textile mills, and worker colonies — includes garment factories, industrial estates, and residential townships.',
            'east': 'Eastern Ludhiana including Dugri area, BRS Nagar colony, educational institutions, and developing zones — includes colleges, housing projects, and commercial centers.',
            'west': 'Western Ludhiana covering Sarabha Nagar area, Ferozepur Road corridor, residential colonies, and connectivity infrastructure — includes housing developments, local markets, and transportation routes.'
        },
        
        # Agra
        'agra': {
            'center': 'Central Agra with Taj Mahal UNESCO site, Agra Fort, Sadar Bazaar, and heritage areas — includes world-famous monuments, traditional markets, and tourism infrastructure.',
            'north': 'Northern Agra covering Sikandra area, Akbar\'s Tomb, residential colonies, and developing zones — includes historical sites, housing projects, and commercial developments.',
            'south': 'Southern Agra featuring industrial areas, leather manufacturing zones, Artoni area, and worker settlements — includes tanneries, handicraft centers, and residential areas.',
            'east': 'Eastern Agra including Dayalbagh area, educational institutions, residential developments, and commercial centers — includes colleges, housing projects, and local markets.',
            'west': 'Western Agra covering Agra Cantonment, military area, upscale residential zones, and connectivity infrastructure — includes defense establishments, modern housing, and transportation hubs.'
        },
        
        # Ghaziabad
        'ghaziabad': {
            'center': 'Central Ghaziabad with Railway Road area, Ghaziabad Junction station, commercial markets, and administrative buildings — includes government offices, business centers, and traditional markets.',
            'north': 'Northern Ghaziabad covering Raj Nagar area, residential townships, educational institutions, and modern developments — includes housing projects, schools, and commercial complexes.',
            'south': 'Southern Ghaziabad featuring industrial areas, Modinagar connection, manufacturing zones, and worker colonies — includes factories, industrial estates, and residential areas.',
            'east': 'Eastern Ghaziabad including Vasundhara area, residential developments, IT parks, and commercial centers — includes modern housing, tech companies, and shopping complexes.',
            'west': 'Western Ghaziabad covering Kaushambi area, metro connectivity, residential projects, and developing infrastructure — includes metro stations, housing developments, and commercial zones.'
        },
        
        # Visakhapatnam
        'visakhapatnam': {
            'center': 'Central Visakhapatnam with RK Beach area, Kailasagiri hill park, Port area, and commercial districts — includes beaches, tourist attractions, and business centers.',
            'north': 'Northern Visakhapatnam covering Rushikonda beach, IT SEZ area, educational institutions, and residential developments — includes tech parks, colleges, and modern housing projects.',
            'south': 'Southern Visakhapatnam featuring steel plant area, industrial zones, Gangavaram port, and manufacturing districts — includes heavy industries, port facilities, and worker townships.',
            'east': 'Eastern Visakhapatnam including coastal areas, fishing villages, beach resorts, and tourism infrastructure — includes beaches, hospitality sector, and coastal communities.',
            'west': 'Western Visakhapatnam covering Madhurawada area, residential projects, educational institutions, and developing zones — includes housing developments, colleges, and commercial centers.'
        },
        
        # Srinagar
        'srinagar': {
            'center': 'Central Srinagar with Dal Lake, Lal Chowk commercial area, Hazratbal Shrine, and heritage sites — includes famous lakes, traditional markets, and cultural centers.',
            'north': 'Northern Srinagar covering Nishat Bagh, Shalimar Bagh, residential areas, and tourist zones — includes Mughal gardens, hospitality infrastructure, and scenic areas.',
            'south': 'Southern Srinagar featuring airport area, Pampore saffron fields, industrial zones, and connectivity infrastructure — includes transportation hubs, agricultural areas, and commercial developments.',
            'east': 'Eastern Srinagar including Nagin Lake area, residential colonies, educational institutions, and developing zones — includes water bodies, housing projects, and colleges.',
            'west': 'Western Srinagar covering Chashme Shahi area, hill stations, residential developments, and scenic locations — includes tourist attractions, housing projects, and natural landscapes.'
        },
        
        # Jammu
        'jammu': {
            'center': 'Central Jammu with Raghunath Temple, Bahu Fort, Jammu Tawi railway station, and commercial areas — includes religious sites, heritage buildings, and business districts.',
            'north': 'Northern Jammu covering Gandhi Nagar area, residential colonies, educational institutions, and developing zones — includes housing projects, colleges, and commercial centers.',
            'south': 'Southern Jammu featuring industrial areas, Bari Brahmana zone, manufacturing districts, and connectivity infrastructure — includes factories, industrial estates, and transportation hubs.',
            'east': 'Eastern Jammu including Channi Himmat area, residential developments, local markets, and community centers — includes housing colonies, traditional markets, and cultural areas.',
            'west': 'Western Jammu covering Trikuta Nagar area, residential projects, educational institutions, and modern developments — includes housing developments, schools, and commercial complexes.'
        }
    }
    
    # Default descriptions for cities not specifically mapped
    default_descriptions = {
        'center': f'Central business district and urban core of {city_name} — featuring government offices, commercial centers, and key administrative buildings.',
        'north': f'Northern region of {city_name} with suburban and residential areas — includes local markets, educational institutions, and community centers.',
        'south': f'Southern region of {city_name} including industrial and residential zones — mix of manufacturing areas, housing developments, and commercial districts.',
        'east': f'Eastern region of {city_name} with mixed urban development — blend of residential neighborhoods, local businesses, and transportation corridors.',
        'west': f'Western region of {city_name} featuring diverse urban landscapes — combination of residential areas, commercial zones, and developing infrastructure.'
    }
    
    return descriptions.get(city_lower, default_descriptions)

def calculate_city_regions(city_bbox: Tuple[float, float, float, float], city_name: str) -> List[Dict]:
    """
    Divide city into 5 regions: North, South, East, West, Center
    
    Args:
        city_bbox: (min_lon, min_lat, max_lon, max_lat)
        city_name: Name of the city
        
    Returns:
        List of region dictionaries with bbox and metadata
    """
    min_lon, min_lat, max_lon, max_lat = city_bbox
    
    # Get city-specific descriptions
    region_descriptions = get_city_specific_descriptions(city_name)
    
    # Calculate city dimensions
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    center_lon = min_lon + lon_range / 2
    center_lat = min_lat + lat_range / 2
    
    # Calculate aspect ratio and adjust region size for better proportions
    aspect_ratio = lon_range / lat_range
    
    # IMPROVED region size calculation to prevent tall/narrow strips
    # Use consistent subdivision approach to maintain better aspect ratios
    
    if aspect_ratio < 0.7:  # Tall, narrow cities like Mumbai
        # For tall cities, use wider regions and shorter height
        region_width = lon_range * 0.4   # Use more of the width
        region_height = lat_range * 0.2  # Use less of the height
    elif aspect_ratio > 1.4:  # Wide cities
        # For wide cities, use taller regions and narrower width
        region_width = lon_range * 0.2   # Use less of the width
        region_height = lat_range * 0.4  # Use more of the height
    else:  # Balanced cities - IMPROVED subdivision
        # Use 2.5 step instead of 3 to prevent overly thin strips
        lon_step = lon_range / 2.5  # More balanced horizontal division
        lat_step = lat_range / 2.5  # More balanced vertical division
        
        # Ensure regions are more square-like
        target_aspect = 1.2  # Slightly rectangular but not extreme
        region_width = min(lon_step, lat_step * target_aspect)
        region_height = min(lat_step, lon_step / target_aspect)
    
    # Ensure minimum region size (at least 0.05 degrees in each dimension)
    region_width = max(region_width, 0.05)
    region_height = max(region_height, 0.05)
    
    # Ensure regions don't exceed city bounds
    region_width = min(region_width, lon_range * 0.45)  # Max 45% of city width
    region_height = min(region_height, lat_range * 0.45)  # Max 45% of city height
    
    # Add padding to prevent regions from going outside city bounds
    padding_lon = lon_range * 0.05
    padding_lat = lat_range * 0.05
    
    # Calculate proper region positions to avoid overlap and stay within bounds
    # Center region - stays in the middle
    center_region = {
        "id": "center",
        "name": f"{city_name} Center",
        "description": region_descriptions['center'],
        "bbox": [
            max(min_lon + padding_lon, center_lon - region_width/2),
            max(min_lat + padding_lat, center_lat - region_height/2),
            min(max_lon - padding_lon, center_lon + region_width/2),
            min(max_lat - padding_lat, center_lat + region_height/2)
        ],
        "coordinates": {
            "center": [center_lat, center_lon],
            "bounds": [
                [center_lat - region_height/2, center_lon - region_width/2],
                [center_lat + region_height/2, center_lon + region_width/2]
            ]
        }
    }
    
    # North region - upper part of the city
    north_start_lat = center_lat + region_height/2 + padding_lat
    north_region = {
        "id": "north",
        "name": f"{city_name} North",
        "description": region_descriptions['north'],
        "bbox": [
            max(min_lon + padding_lon, center_lon - region_width/2),
            north_start_lat,
            min(max_lon - padding_lon, center_lon + region_width/2),
            min(max_lat - padding_lat, north_start_lat + region_height)
        ],
        "coordinates": {
            "center": [north_start_lat + region_height/2, center_lon],
            "bounds": [
                [north_start_lat, center_lon - region_width/2],
                [north_start_lat + region_height, center_lon + region_width/2]
            ]
        }
    }
    
    # South region - lower part of the city
    south_end_lat = center_lat - region_height/2 - padding_lat
    south_region = {
        "id": "south",
        "name": f"{city_name} South",
        "description": region_descriptions['south'],
        "bbox": [
            max(min_lon + padding_lon, center_lon - region_width/2),
            max(min_lat + padding_lat, south_end_lat - region_height),
            min(max_lon - padding_lon, center_lon + region_width/2),
            south_end_lat
        ],
        "coordinates": {
            "center": [south_end_lat - region_height/2, center_lon],
            "bounds": [
                [south_end_lat - region_height, center_lon - region_width/2],
                [south_end_lat, center_lon + region_width/2]
            ]
        }
    }
    
    # For east/west regions, adjust dimensions to maintain good aspect ratios
    # Calculate available space for east/west regions
    available_east_width = (max_lon - padding_lon) - (center_lon + region_width/2 + padding_lon)
    available_west_width = (center_lon - region_width/2 - padding_lon) - (min_lon + padding_lon)
    
    # Use the smaller available width to ensure both regions fit
    side_region_width = min(available_east_width, available_west_width, region_width)
    
    # For tall cities like Mumbai, make east/west regions more square
    if aspect_ratio < 0.7:  # Tall, narrow cities
        side_region_height = min(region_height * 0.8, side_region_width * 1.2)  # More square proportions
    else:
        side_region_height = region_height
    
    # East region - right side of the city
    east_start_lon = center_lon + region_width/2 + padding_lon
    east_region = {
        "id": "east",
        "name": f"{city_name} East",
        "description": region_descriptions['east'],
        "bbox": [
            east_start_lon,
            max(min_lat + padding_lat, center_lat - side_region_height/2),
            min(max_lon - padding_lon, east_start_lon + side_region_width),
            min(max_lat - padding_lat, center_lat + side_region_height/2)
        ],
        "coordinates": {
            "center": [center_lat, east_start_lon + side_region_width/2],
            "bounds": [
                [center_lat - side_region_height/2, east_start_lon],
                [center_lat + side_region_height/2, east_start_lon + side_region_width]
            ]
        }
    }
    
    # West region - left side of the city
    west_end_lon = center_lon - region_width/2 - padding_lon
    west_region = {
        "id": "west",
        "name": f"{city_name} West",
        "description": region_descriptions['west'],
        "bbox": [
            max(min_lon + padding_lon, west_end_lon - side_region_width),
            max(min_lat + padding_lat, center_lat - side_region_height/2),
            west_end_lon,
            min(max_lat - padding_lat, center_lat + side_region_height/2)
        ],
        "coordinates": {
            "center": [center_lat, west_end_lon - side_region_width/2],
            "bounds": [
                [center_lat - side_region_height/2, west_end_lon - side_region_width],
                [center_lat + side_region_height/2, west_end_lon]
            ]
        }
    }
    
    regions = [center_region, north_region, south_region, east_region, west_region]
    
    return regions

async def generate_region_preview(region_bbox: Tuple[float, float, float, float], 
                                region_name: str) -> Optional[str]:
    """
    Generate real satellite preview image using Google Earth Engine
    
    Args:
        region_bbox: Region bounding box (min_lon, min_lat, max_lon, max_lat)
        region_name: Name of the region
        
    Returns:
        URL path to the generated preview image
    """
    try:
        logger.info(f"Generating real satellite preview for {region_name}")
        
        # Use Google Earth Engine to fetch real satellite imagery
        if gee_fetcher.initialized:
            # Fetch real satellite data using GEE
            satellite_data = gee_fetcher.fetch_satellite_data(region_bbox)
            
            if satellite_data and satellite_data.get('ndvi_array') is not None:
                from datetime import datetime
                import matplotlib.pyplot as plt
                from pathlib import Path
                import ee
                
                # Create preview directory
                preview_dir = Path("./outputs/region_previews")
                preview_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = region_name.replace(' ', '_').replace('-', '_')
                filename = f"satellite_{safe_name}_{timestamp}.png"
                filepath = preview_dir / filename
                
                # Get real RGB satellite image from Google Earth Engine
                try:
                    # Create geometry for the region
                    geometry = ee.Geometry.Rectangle([
                        region_bbox[0], region_bbox[1], 
                        region_bbox[2], region_bbox[3]
                    ])
                    
                    # Get Sentinel-2 imagery (high resolution, true color)
                    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                        .filterBounds(geometry) \
                        .filterDate('2023-01-01', '2024-12-31') \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                        .select(['B4', 'B3', 'B2'])  # Red, Green, Blue bands
                    
                    # Get the most recent cloud-free image
                    image = sentinel2.median().clip(geometry)
                    
                    # Scale values for visualization (Sentinel-2 values are 0-10000)
                    rgb_image = image.divide(10000).multiply(255)
                    
                    # Get the image as array
                    rgb_array = rgb_image.sampleRectangle(geometry, defaultValue=0)
                    
                    # Extract RGB bands
                    red_data = np.array(rgb_array.get('B4').getInfo())
                    green_data = np.array(rgb_array.get('B3').getInfo()) 
                    blue_data = np.array(rgb_array.get('B2').getInfo())
                    
                    # Stack RGB channels
                    rgb_stack = np.stack([red_data, green_data, blue_data], axis=-1)
                    rgb_stack = np.clip(rgb_stack, 0, 255).astype(np.uint8)
                    
                    # Create the satellite preview
                    plt.figure(figsize=(10, 8), dpi=150)
                    plt.imshow(rgb_stack)
                    plt.axis('off')
                    plt.title(f'{region_name} - Real Satellite Imagery', 
                             fontsize=14, fontweight='bold', pad=15)
                    
                    # Add data source info
                    plt.figtext(0.02, 0.02, 'Source: Sentinel-2 via Google Earth Engine', 
                               fontsize=8, style='italic', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(filepath, bbox_inches='tight', dpi=150, 
                               facecolor='white', edgecolor='none')
                    plt.close()
                    
                    logger.info(f"Real satellite preview saved: {filename}")
                    return f"/outputs/region_previews/{filename}"
                    
                except Exception as gee_error:
                    logger.warning(f"GEE satellite fetch failed: {gee_error}, using NDVI visualization")
                    
                    # Fallback: Use NDVI data to create a vegetation map
                    plt.figure(figsize=(10, 8), dpi=150)
                    
                    # Create NDVI visualization (green = high vegetation, brown = low vegetation)
                    ndvi_data = satellite_data['ndvi_array']
                    
                    # Create custom colormap for vegetation
                    from matplotlib.colors import LinearSegmentedColormap
                    colors = ['#8B4513', '#D2B48C', '#FFFF99', '#90EE90', '#228B22', '#006400']  # Brown to dark green
                    n_bins = 100
                    vegetation_cmap = LinearSegmentedColormap.from_list('vegetation', colors, N=n_bins)
                    
                    plt.imshow(ndvi_data, cmap=vegetation_cmap, vmin=0, vmax=1)
                    plt.colorbar(label='NDVI (Vegetation Index)', shrink=0.8)
                    plt.axis('off')
                    plt.title(f'{region_name} - Vegetation Analysis', 
                             fontsize=14, fontweight='bold', pad=15)
                    
                    # Add data source info
                    plt.figtext(0.02, 0.02, 'Source: MODIS NDVI via Google Earth Engine', 
                               fontsize=8, style='italic', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(filepath, bbox_inches='tight', dpi=150, 
                               facecolor='white', edgecolor='none')
                    plt.close()
                    
                    logger.info(f"NDVI vegetation preview saved: {filename}")
                    return f"/outputs/region_previews/{filename}"
            
        else:
            logger.warning("Google Earth Engine not initialized, creating basic preview")
            
            # Create a simple informational preview when GEE is not available
            from datetime import datetime
            import matplotlib.pyplot as plt
            from pathlib import Path
            
            preview_dir = Path("./outputs/region_previews")
            preview_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = region_name.replace(' ', '_').replace('-', '_')
            filename = f"info_{safe_name}_{timestamp}.png"
            filepath = preview_dir / filename
            
            # Create informational preview
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            ax.text(0.5, 0.6, f'{region_name}', ha='center', va='center', 
                   fontsize=20, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.4, 'Satellite imagery will be\ngenerated during analysis', 
                   ha='center', va='center', fontsize=12, 
                   transform=ax.transAxes, style='italic')
            ax.text(0.5, 0.2, f'Coordinates: {region_bbox[1]:.3f}°N, {region_bbox[0]:.3f}°E', 
                   ha='center', va='center', fontsize=10, 
                   transform=ax.transAxes, alpha=0.7)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#f0f8ff')
            
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight', dpi=100, facecolor='white')
            plt.close()
            
            logger.info(f"Info preview saved: {filename}")
            return f"/outputs/region_previews/{filename}"
            
    except Exception as e:
        logger.error(f"Failed to generate preview for {region_name}: {e}")
        return None

def get_fallback_city_bbox(city_name: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Fallback city coordinates for common Indian cities
    Returns: (min_lon, min_lat, max_lon, max_lat)
    """
    fallback_cities = {
        # 28 State Capitals
        'mumbai': (72.7760, 18.8900, 72.9800, 19.2700),           # Maharashtra
        'bangalore': (77.4600, 12.8340, 77.7800, 13.1390),        # Karnataka
        'bengaluru': (77.4600, 12.8340, 77.7800, 13.1390),        # Karnataka (alternate name)
        'chennai': (80.0950, 12.8340, 80.3200, 13.2340),          # Tamil Nadu
        'hyderabad': (78.2580, 17.2540, 78.6370, 17.5650),        # Telangana
        'kolkata': (88.2640, 22.4640, 88.4340, 22.6500),          # West Bengal
        'ahmedabad': (72.4580, 22.9340, 72.7580, 23.1340),        # Gujarat (largest city)
        'gandhinagar': (72.5369, 23.1156, 72.7369, 23.3156),      # Gujarat (official capital)
        'jaipur': (75.6390, 26.8340, 75.9390, 27.0340),           # Rajasthan
        'lucknow': (80.7390, 26.6340, 81.0390, 26.9340),          # Uttar Pradesh
        'bhopal': (77.2390, 23.1340, 77.5390, 23.3340),           # Madhya Pradesh
        'patna': (85.0390, 25.5340, 85.2390, 25.7340),            # Bihar
        'thiruvananthapuram': (76.8366, 8.3855, 77.0866, 8.6855), # Kerala
        'bhubaneswar': (85.6985, 20.1461, 85.9985, 20.4461),      # Odisha
        'ranchi': (85.1722, 23.2441, 85.4722, 23.5441),           # Jharkhand
        'raipur': (81.4615, 21.1514, 81.7615, 21.4514),           # Chhattisgarh
        'panaji': (73.7278, 15.3909, 73.9278, 15.5909),           # Goa
        'shimla': (77.0734, 31.0048, 77.3734, 31.3048),           # Himachal Pradesh
        'srinagar': (74.6973, 33.9837, 74.9973, 34.2837),         # Jammu & Kashmir
        'jammu': (74.7570, 32.6266, 75.0570, 32.9266),            # Jammu & Kashmir (winter capital)
        'guwahati': (91.6362, 26.0445, 91.9362, 26.3445),         # Assam (largest city)
        'dispur': (91.6898, 26.0433, 91.9898, 26.3433),           # Assam (official capital)
        'agartala': (91.1868, 23.7315, 91.4868, 24.0315),         # Tripura
        'aizawl': (92.6173, 23.6271, 92.9173, 23.9271),           # Mizoram
        'imphal': (93.8063, 24.7170, 94.1063, 25.0170),           # Manipur
        'kohima': (94.0086, 25.5751, 94.3086, 25.8751),           # Nagaland
        'itanagar': (93.5053, 26.9844, 93.8053, 27.2844),         # Arunachal Pradesh
        'gangtok': (88.5138, 27.2389, 88.8138, 27.5389),          # Sikkim
        'shillong': (91.7933, 25.4788, 92.0933, 25.7788),         # Meghalaya
        
        # 8 Union Territory Capitals
        'delhi': (76.8380, 28.4040, 77.3480, 28.8840),            # Delhi
        'new delhi': (76.8380, 28.4040, 77.3480, 28.8840),        # Delhi (alternate name)
        'chandigarh': (76.6794, 30.6333, 76.9794, 30.9333),       # Chandigarh & Punjab & Haryana
        'puducherry': (79.7083, 11.8139, 80.0083, 12.1139),       # Puducherry
        'port blair': (92.6265, 11.5234, 92.9265, 11.8234),       # Andaman & Nicobar
        'kavaratti': (72.5369, 10.4669, 72.8369, 10.7669),        # Lakshadweep
        'daman': (72.7397, 20.2974, 73.0397, 20.5974),            # Daman & Diu
        'silvassa': (72.9169, 20.1738, 73.2169, 20.4738),         # Dadra & Nagar Haveli
        'ladakh': (77.4771, 34.0526, 77.7771, 34.3526),           # Ladakh (Leh)
        
        # Major cities for comprehensive testing
        'pune': (73.7390, 18.4290, 73.9890, 18.6390),             # Maharashtra
        'nagpur': (79.0390, 21.0340, 79.2390, 21.2340),           # Maharashtra
        'indore': (75.6390, 22.6340, 75.9390, 22.8340),           # Madhya Pradesh
        'kanpur': (80.1390, 26.3340, 80.4390, 26.5340),           # Uttar Pradesh
        'thane': (72.9390, 19.1340, 73.1390, 19.3340),            # Maharashtra
        'visakhapatnam': (83.1390, 17.5340, 83.4390, 17.8340),    # Andhra Pradesh
        'pimpri chinchwad': (73.6390, 18.5340, 73.9390, 18.7340), # Maharashtra
        
        # Additional major cities
        'surat': (72.7811, 21.1702, 72.8811, 21.2702),            # Gujarat
        'vadodara': (73.1812, 22.3072, 73.2812, 22.4072),         # Gujarat
        'rajkot': (70.7722, 22.3039, 70.8722, 22.4039),           # Gujarat
        'faridabad': (77.2975, 28.4089, 77.3975, 28.5089),        # Haryana
        'ghaziabad': (77.4126, 28.6692, 77.5126, 28.7692),        # Uttar Pradesh
        'ludhiana': (75.8573, 30.9010, 75.9573, 31.0010),         # Punjab
        'agra': (78.0081, 27.1767, 78.1081, 27.2767),              # Uttar Pradesh
        'nashik': (73.7898, 19.9975, 73.8898, 20.0975),           # Maharashtra
        'meerut': (77.7064, 28.9845, 77.8064, 29.0845),           # Uttar Pradesh
        'kalyan': (73.1645, 19.2403, 73.2645, 19.3403),           # Maharashtra
        'vasai virar': (72.8397, 19.3919, 72.9397, 19.4919),      # Maharashtra
        'varanasi': (82.9739, 25.3176, 83.0739, 25.4176),         # Uttar Pradesh
        'aurangabad': (75.3433, 19.8762, 75.4433, 19.9762),       # Maharashtra
        'dhanbad': (86.4304, 23.7957, 86.5304, 23.8957),          # Jharkhand
        'amritsar': (74.8723, 31.6340, 74.9723, 31.7340),         # Punjab
        'navi mumbai': (73.0297, 19.0330, 73.1297, 19.1330),      # Maharashtra
        'allahabad': (81.8463, 25.4358, 81.9463, 25.5358),        # Uttar Pradesh
        'howrah': (88.2636, 22.5958, 88.3636, 22.6958),           # West Bengal
        'coimbatore': (76.9558, 11.0168, 77.0558, 11.1168),       # Tamil Nadu
        'jabalpur': (79.9864, 23.1815, 80.0864, 23.2815),         # Madhya Pradesh
        'gwalior': (78.1828, 26.2183, 78.2828, 26.3183),          # Madhya Pradesh
        'vijayawada': (80.6480, 16.5062, 80.7480, 16.6062),       # Andhra Pradesh
        'jodhpur': (73.0243, 26.2389, 73.1243, 26.3389),          # Rajasthan
        'madurai': (78.1198, 9.9252, 78.2198, 10.0252),           # Tamil Nadu
        'kota': (75.8648, 25.2138, 75.9648, 25.3138),             # Rajasthan
        'solapur': (75.9064, 17.6599, 76.0064, 17.7599)           # Maharashtra
    }
    
    city_lower = city_name.lower().strip()
    return fallback_cities.get(city_lower)

@router.post("/get-city-regions", response_model=RegionResponse)
async def get_city_regions(request: RegionRequest):
    """
    Get available regions for a city with satellite previews
    """
    try:
        logger.info(f"Getting regions for city: {request.city}")
        
        # Step 1: Get city bounding box (try geocoding first, then fallback)
        city_bbox = gee_fetcher.get_city_bbox(request.city)
        
        if not city_bbox:
            logger.warning(f"Geocoding failed for {request.city}, trying fallback coordinates")
            city_bbox = get_fallback_city_bbox(request.city)
        
        if not city_bbox:
            raise HTTPException(
                status_code=404,
                detail=f"City '{request.city}' not found. Please check the spelling and try again."
            )
        
        # Step 2: Calculate regions
        regions_data = calculate_city_regions(city_bbox, request.city)
        
        # Step 3: Use cached satellite images but don't show them immediately
        # Frontend will fake the loading process for better UX
        from app.api.cache_service import cache_service
        
        regions_with_previews = []
        for region_data in regions_data:
            # Check if cached satellite image exists
            preview_url = None
            if cache_service.is_city_cached(request.city):
                # Use cached satellite image (but frontend will fake loading)
                preview_url = f"/api/cached-image/{request.city.replace(' ', '_')}/{region_data['id']}"
                logger.info(f"Cached image available for {region_data['name']}: {preview_url}")
            else:
                logger.info(f"No cached image for {region_data['name']}")
            
            # Create region info
            region_info = RegionInfo(
                id=region_data['id'],
                name=region_data['name'],
                description=region_data['description'],
                bbox=region_data['bbox'],
                coordinates=region_data['coordinates'],
                preview_image_url=preview_url  # Cached image URL (frontend will fake loading)
            )
            regions_with_previews.append(region_info)
        
        # Step 4: Calculate city center
        min_lon, min_lat, max_lon, max_lat = city_bbox
        city_center = [
            min_lat + (max_lat - min_lat) / 2,  # center_lat
            min_lon + (max_lon - min_lon) / 2   # center_lon
        ]
        
        response = RegionResponse(
            city=request.city,
            total_regions=len(regions_with_previews),
            regions=regions_with_previews,
            city_center=city_center,
            city_bbox=list(city_bbox)
        )
        
        logger.info(f"Successfully generated {len(regions_with_previews)} regions for {request.city}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting city regions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get city regions: {str(e)}"
        )

class RegionAnalysisRequest(BaseModel):
    """Model for region analysis request"""
    region_bbox: List[float]
    region_name: str
    city: str

@router.post("/analyze-region")
async def analyze_specific_region(request: RegionAnalysisRequest):
    """
    Analyze a specific region for biomass prediction
    This endpoint will be called after user selects a region
    """
    try:
        logger.info(f"Analyzing region: {request.region_name} in {request.city}")
        
        # Convert to tuple for existing functions
        bbox_tuple = tuple(request.region_bbox)
        
        # Step 1: Fetch satellite data for the region
        satellite_data = gee_fetcher.fetch_satellite_data(bbox_tuple)
        if not satellite_data:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch satellite data for region"
            )
        
        # Step 2: Calculate biomass from satellite indices
        biomass_results = gee_fetcher.calculate_biomass_from_indices(
            satellite_data['ndvi'],
            satellite_data['evi'],
            satellite_data['lai']
        )
        
        # Step 3: Generate biomass forecasting
        forecast_data = gee_fetcher.forecast_biomass(
            biomass_results['total_agb'],
            satellite_data['ndvi'],
            satellite_data['lai']
        )
        
        # Step 4: Calculate urban performance metrics
        def calculate_urban_metrics(agb: float, canopy_cover: float):
            epi_score = int(min(100, (agb / 100.0) * 60 + (canopy_cover / 100) * 40))
            tree_cities_score = int(min(100, (canopy_cover / 25.0) * 100))
            green_space_ratio = round(canopy_cover / 100.0, 3)
            
            return {
                "epi_score": epi_score,
                "tree_cities_score": tree_cities_score,
                "green_space_ratio": green_space_ratio
            }
        
        urban_metrics = calculate_urban_metrics(
            biomass_results['total_agb'],
            biomass_results['canopy_cover']
        )
        
        # Step 5: Generate planning recommendations
        def generate_recommendations(agb: float, canopy_cover: float, epi_score: int):
            recommendations = []
            
            if canopy_cover < 25:
                recommendations.append(
                    f"Current canopy cover is {canopy_cover:.1f}%. Consider establishing "
                    "urban forests and green corridors to increase carbon sequestration capacity."
                )
            
            if agb < 40:
                recommendations.append(
                    "Current biomass density is below optimal levels. Focus on: "
                    "native species plantations, green roof programs, and park expansion."
                )
            
            if epi_score < 60:
                recommendations.append(
                    f"EPI score ({epi_score}/100) indicates room for improvement. "
                    "Implement smart irrigation systems to maintain vegetation health "
                    "during dry seasons and maximize biomass growth potential."
                )
            
            recommendations.append(
                "Create neighborhood-level green infrastructure plans to distribute "
                "biomass equitably across all districts, especially in high-density areas."
            )
            
            if agb > 70:
                recommendations.append(
                    f"Excellent biomass density ({agb:.1f} Mg/ha)! Maintain current "
                    "green spaces and consider implementing a monitoring program."
                )
            
            return recommendations[:5]
        
        recommendations = generate_recommendations(
            biomass_results['total_agb'],
            biomass_results['canopy_cover'],
            urban_metrics['epi_score']
        )
        
        # Step 6: Generate visualization heatmap
        from app.api.satellite_image_generator import generate_satellite_heatmap
        from datetime import datetime
        
        heatmap_path = generate_satellite_heatmap(
            request.region_name,
            f"Regional Analysis - {request.city}",
            satellite_data['ndvi_array'],
            biomass_results,
            bbox_tuple,
            use_real_satellite=True
        )
        
        # Step 7: Calculate region center coordinates
        min_lon, min_lat, max_lon, max_lat = bbox_tuple
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        # Step 8: Prepare comprehensive response
        response = {
            "city": request.city,
            "region_name": request.region_name,
            "location": {
                "coordinates": f"{center_lat:.4f}, {center_lon:.4f}",
                "bbox": list(bbox_tuple)
            },
            "timestamp": datetime.now().isoformat(),
            "satellite_data": {
                "ndvi": satellite_data['ndvi'],
                "evi": satellite_data['evi'],
                "lai": satellite_data['lai'],
                "lst": satellite_data['lst'],
                "data_source": satellite_data['data_source']
            },
            "current_agb": biomass_results,
            "forecasting": forecast_data,
            "urban_metrics": urban_metrics,
            "planning_recommendations": recommendations,
            "heat_map": {
                "image_url": f"http://localhost:8000{heatmap_path}",
                "description": f"Biomass analysis for {request.region_name}, {request.city}"
            }
        }
        
        logger.info(f"Regional analysis completed successfully for {request.region_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing region: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze region: {str(e)}"
        )

