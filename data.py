import pandas as pd
import numpy as np

# Initial data trimming
data_dtypes = {
    'state': str,
    'n_killed': np.uint16,
    'n_injured': np.uint16,
    'congressional_district': np.uint8,
}

data = pd.read_csv(
    'gun_violence.csv',
    parse_dates=['date']
)

idx = ~data.congressional_district.isna()
idx &= ~data.participant_age.isna()
idx &= ~data.participant_gender.isna()
idx &= ~data.incident_characteristics.isna()
idx &= ~data.latitude.isna()
idx &= ~data.longitude.isna()
data = data[idx]

for dtype in data_dtypes:
    data[dtype] = data[dtype].apply(data_dtypes[dtype])

data.drop([
    'n_guns_involved',
    'participant_age_group',
    'participant_name',
    'participant_relationship',
    'participant_status',
    'location_description',
    'city_or_county',
    'incident_url_fields_missing',
    'incident_url',
    'incident_id',
    'address',
    'sources',
    'source_url',
    'notes',
    'state_house_district',
    'state_senate_district',
    'gun_stolen',
    'gun_type'], axis=1, inplace=True)

# Column parsing
def to_dict(x):
    di = {}
    x = x.replace('||', '|')
    x = x.replace('::', ':')
    x = x.split('|')
    for string in x:
        li = string.split(':')
        di.update({int(li[0]): li[1].split(',')})
    return di

def to_set(x):
    x = x.replace('||', '|')
    return set(x.split('|'))

def participant_mask(x):
    l0 = len(x['participant_age'])
    l1 = len(x['participant_type'])
    l2 = len(x['participant_gender'])
    return l0 == l1 == l2

data['incident_characteristics'] = data.incident_characteristics.apply(to_set)
data['participant_gender'] = data.participant_gender.apply(to_dict)
data['participant_age'] = data.participant_age.apply(to_dict)
data['participant_type'] = data.participant_type.apply(to_dict)
data = data[data.apply(participant_mask, axis=1)]
data.reset_index(drop=True, inplace=True)

# Incident characteristics reduction functions
def drop_chars(incident_chars, drop_chars):
    for drop_char in drop_chars:
        incident_chars.discard(drop_char)
    return incident_chars

def combine_chars(incident_chars, com_chars, new_label):
    for char in com_chars:
        if char in incident_chars:
            incident_chars.discard(char)
            incident_chars.add(new_label)
    return incident_chars

def search_mask(incident_chars, search_chars):
    mask = True
    for char in search_chars:
        mask &= char in incident_chars
    return mask

def ignore_mask(incident_chars, ignore_chars):
    mask = True
    for char in ignore_chars:
        mask &= char not in incident_chars
    return mask

def add_char(incident_chars, label):
    incident_chars.add(label)
    return incident_chars

def num_chars(incident_chars):
    return len(incident_chars)

# May not need ******************************************
chars_before = []
for el in data.incident_characteristics:
    chars_before += el
    
chars_before = set(chars_before)
char_li_before = list(chars_before)

freq_mtx = np.zeros((len(chars_before), len(chars_before)), dtype=np.int32)

for char in chars_before:
    for row in data.incident_characteristics:
        if char in row:
            for el in row:
                freq_mtx[char_li_before.index(char), char_li_before.index(el)] += 1

freq_df = pd.DataFrame(freq_mtx)

di = {i: char for i, char in enumerate(char_li_before)}
freq_df.index = char_li_before
freq_df.rename(columns=di, inplace=True)
# *********************************************************

# Incident characteristics reduction and simplification process
# ------------------------------------------------------------------------------
char_drop_li = [
    'Accidental Shooting - Death',
    'Accidental Shooting - Injury',
    'Accidental Shooting at a Business',
    'Accidental/Negligent Discharge',
    'Child Involved Incident',
    'Child injured (not child shooter)',
    'Child injured by child',
    'Child injured self',
    'Child killed (not child shooter)',
    'Child killed by child',
    'Child killed self',
    'Child picked up & fired gun',
    'Child with gun - no shots fired',
    'Defensive Use - Crime occurs, victim shoots subject/suspect/perpetrator',
    'Defensive Use - Good Samaritan/Third Party',
    'Defensive Use - Shots fired, no injury/death',
    'Defensive Use - Stand Your Ground/Castle Doctrine established',
    'Defensive Use - Victim stops crime',
    'Defensive Use - WITHOUT a gun',
    'Defensive use - No shots fired',
    'Home Invasion - No death or injury',
    'Home Invasion - Resident injured',
    'Home Invasion - Resident killed',
    'Home Invasion - subject/suspect/perpetrator injured',
    'Home Invasion - subject/suspect/perpetrator killed',
    'Officer Involved Incident - Weapon involved but no shots fired',
    'Officer Involved Shooting - Accidental discharge - no injury required',
    'Officer Involved Shooting - Bystander killed',
    'Officer Involved Shooting - Bystander shot',
    'Officer Involved Shooting - Officer killed',
    'Officer Involved Shooting - Officer shot',
    'Officer Involved Shooting - Shots fired, no injury',
    'Officer Involved Shooting - subject/suspect/perpetrator killed',
    'Officer Involved Shooting - subject/suspect/perpetrator shot',
    'Officer Involved Shooting - subject/suspect/perpetrator suicide at standoff',
    'Officer Involved Shooting - subject/suspect/perpetrator suicide by cop',
    'Officer Involved Shooting - subject/suspect/perpetrator surrender at standoff',
    'Officer Involved Shooting - subject/suspect/perpetrator unarmed',
    'Hunting accident',
    'NAV',
    'Playing with gun',
    'Mistaken ID (thought it was an intruder/threat, was friend/family)',
    'Cleaning gun',
    'Gun buy back action',
    'Political Violence',
    'School Shooting - elementary/secondary school',
    'School Shooting - university/college',
    'Gun at school, no death/injury - elementary/secondary school',
    'Gun at school, no death/injury - university/college',
    'ShotSpotter',
    'Institution/Group/Business',
    'Police Targeted',
    'Terrorism Involvement',
    'Hate crime',
    'Assault weapon (AR-15, AK-47, and ALL variants defined by law enforcement)',
    'Workplace shooting (disgruntled employee)',
    'Implied Weapon',
    'Gun range/gun shop/gun show shooting',
    'Thought gun was unloaded',
    'Self-Inflicted (not suicide or suicide attempt - NO PERP)',
    'Shot - Dead (murder, accidental, suicide)',
    'Shot - Wounded/Injured',
    'Shots Fired - No Injuries',
    'Shots fired, no action (reported, no evidence found)',
    'Animal shot/killed'
]

# Drop unnecessary characteristics
data.incident_characteristics = data.incident_characteristics.apply(
    drop_chars,
    drop_chars=char_drop_li)

# Rename drive-by
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['Drive-by (car to street, car to car)'],
    new_label='Drive-by')

# Rename shootout
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['Shootout (where VENN diagram of shooters and victims overlap)'],
    new_label='Shootout')

# Rename spree shooting
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['Spree Shooting (multiple victims, multiple locations)'],
    new_label='Spree Shooting')
    
# Attempted murder/suicide incidents classified as murder/suicide incidents
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['Attempted Murder/Suicide (one variable unsuccessful)'],
    new_label='Murder/Suicide')

# Rename suicide and attempted suicides classified as suicides
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['Suicide^', 'Suicide - Attempt'],
    new_label='Suicide')
mask = data.incident_characteristics.apply(
    search_mask, 
    search_chars=['Murder/Suicide', 'Suicide'])
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    drop_chars,
    drop_chars=['Suicide'])

# Combine characteristics which are mass shootings
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=[
        'Mass Murder (4+ deceased victims excluding the subject/suspect/perpetrator , one location)',
        'Mass Shooting (4+ victims injured or killed excluding the subject/suspect/perpetrator, one location)',
        'Spree Shooting'],
    new_label='Mass/Spree Shooting')

# Combine and simplify characteristics to create new characteristic for concealed carry incident
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars, 
    com_chars=[
        'Concealed Carry License - Perpetrator', 
        'Concealed Carry License - Victim'],
    new_label='Concealed Carry Incident')

# Combine and simplify characteristics to create new characteristic for drug/alcohol involvement
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=[
        'Drug involvement',
        'Under the influence of alcohol or drugs (only applies to the subject/suspect/perpetrator )',
        'House party',
        'Bar/club incident - in or around establishment'],
    new_label='Drug/Alcohol Involvement')

# Combine and simplify characteristics to create new characteristic for non-shooting incident
mask = data.incident_characteristics.apply(
    search_mask,
    search_chars=['Pistol-whipping'])
mask |= data.incident_characteristics.apply(
    search_mask, 
    search_chars=['Brandishing/flourishing/open carry/lost/found'])
mask &= data.incident_characteristics.apply(
    ignore_mask, 
    ignore_chars=['Non-Shooting Incident'])
mask &= data.incident_characteristics.apply(num_chars) != 1
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    drop_chars, drop_chars=[
        'Pistol-whipping',
        'Brandishing/flourishing/open carry/lost/found'])
mask = data.incident_characteristics.apply(
    search_mask, 
    search_chars=['Pistol-whipping'])
mask |= data.incident_characteristics.apply(
    search_mask, 
    search_chars=['Brandishing/flourishing/open carry/lost/found'])
mask &= data.incident_characteristics.apply(num_chars) == 1
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    combine_chars,
    com_chars=[
        'Pistol-whipping',
        'Brandishing/flourishing/open carry/lost/found'], 
    new_label='Non-Shooting Incident')
data.incident_characteristics = data.incident_characteristics.apply(
    drop_chars,
    drop_chars=[
        'Pistol-whipping',
        'Brandishing/flourishing/open carry/lost/found'])
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['LOCKDOWN/ALERT ONLY: No GV Incident Occurred Onsite'],
    new_label='Non-Shooting Incident')

# Combine and simplify characteristics to create new characteristic for guns involved with other crimes
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=[
        'Road rage',
        'Sex crime involving firearm',
        'Kidnapping/abductions/hostage',
        'Home Invasion',
        'Domestic Violence',
        'Possession (gun(s) found during commission of other crimes)',
        'Car-jacking',
        'Gun shop robbery or burglary'],
    new_label='Gun(s) Involved with Other Crimes')
mask = data.incident_characteristics.apply(
    search_mask,
    search_chars=['Armed robbery with injury/death and/or evidence of DGU found'])
mask |= data.incident_characteristics.apply(
    search_mask,
    search_chars=['Criminal act with stolen gun'])
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    add_char,
    label='Gun(s) Involved with Other Crimes')
mask = data.incident_characteristics.apply(
    search_mask,
    search_chars=['Stolen/Illegally owned gun{s} recovered during arrest/warrant'])
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    add_char,
    label='Gun(s) Involved with Other Crimes')

# Combine and simplify characteristics to create new characteristic for unlawful possession of gun
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=[
        'ATF/LE Confiscation/Raid/Arrest',
        'Possession of gun by felon or prohibited person',
        'Unlawful purchase/sale',
        'Ghost gun'],
    new_label='Unlawful Possession of Gun')
mask = data.incident_characteristics.apply(
    search_mask,
    search_chars=['TSA Action'])
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    combine_chars,
    com_chars=['TSA Action'],
    new_label='Unlawful Possession of Gun')
data.incident_characteristics = data.incident_characteristics.apply(
    drop_chars,
    drop_chars=['TSA Action'])
mask = data.incident_characteristics.apply(
    search_mask,
    search_chars=['Criminal act with stolen gun'])
mask |= data.incident_characteristics.apply(
    search_mask,
    search_chars=['Gun(s) stolen from owner'])
mask |= data.incident_characteristics.apply(
    search_mask,
    search_chars=['Guns stolen from law enforcement'])
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    add_char,
    label='Unlawful Possession of Gun')
data.incident_characteristics = data.incident_characteristics.apply(
    combine_chars,
    com_chars=['Stolen/Illegally owned gun{s} recovered during arrest/warrant'],
    new_label='Unlawful Possession of Gun')

mask = data.incident_characteristics.apply(
    search_mask,
    search_chars=[
        'Defensive Use',
        'Armed robbery with injury/death and/or evidence of DGU found'])
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(
    drop_chars,
    drop_chars=['Armed robbery with injury/death and/or evidence of DGU found'])

# Drop characteristics that have been incorporated into new or other characteristics
data.incident_characteristics = data.incident_characteristics.apply(
    drop_chars,
    drop_chars=[
        'Guns stolen from law enforcement',
        'Gun(s) stolen from owner',
        'Criminal act with stolen gun',
        'Armed robbery with injury/death and/or evidence of DGU found'])
# ------------------------------------------------------------------------------

# Incident types
# May not need *****************************************************************
incident_types = [
    'Accidental Shooting',
    'Suicide',
    'Murder/Suicide',
    'Mass/Spree Shooting',
    'Non-Shooting Incident',
]

for inc in incident_types:
    if inc != 'Non-Shooting Incident':
        mask = data.incident_characteristics.apply(search_mask, search_chars=['Non-Shooting Incident', inc])
        data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(drop_chars, drop_chars=['Non-Shooting Incident'])
    if inc != 'Accidental Shooting':
        mask = data.incident_characteristics.apply(search_mask, search_chars=['Accidental Shooting', inc])
        data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(drop_chars, drop_chars=[inc])
    if inc != 'Suicide':
        mask = data.incident_characteristics.apply(search_mask, search_chars=['Suicide', inc])
        data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(drop_chars, drop_chars=['Suicide'])
    if inc != 'Murder/Suicide':
        mask = data.incident_characteristics.apply(search_mask, search_chars=['Murder/Suicide', inc])
        data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(drop_chars, drop_chars=['Murder/Suicide'])
    if inc != 'Shootout':
        mask = data.incident_characteristics.apply(search_mask, search_chars=['Shootout', inc])
        data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(drop_chars, drop_chars=['Shootout'])
    if inc != 'Mass/Spree Shooting':
        mask = data.incident_characteristics.apply(search_mask, search_chars=['Mass/Spree Shooting', inc])
        data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(drop_chars, drop_chars=[inc])
        
mask = data.incident_characteristics.apply(ignore_mask, ignore_chars=incident_types)
data.incident_characteristics[mask] = data.incident_characteristics[mask].apply(add_char, label='Regular Shooting')
incident_types.append('Regular Shooting')

mask = data.incident_characteristics.apply(ignore_mask, ignore_chars=['Shootout'])
data = data[mask]
data = data.reset_index(drop=True)

chars = []
for el in data.incident_characteristics:
    chars += el
    
chars = set(chars)

incident_types = set(incident_types)
incident_descriptors = chars-incident_types
# ******************************************************************************

def bin_column(participants, bins):
    bins = bins.copy()
    for participant in participants:
        bins[participants[participant][0]] += 1
    return bins

genders = {'Male':0, 'Female':0}
cols = pd.DataFrame(list(data.participant_gender.apply(bin_column, bins=genders)))
data = pd.concat([data, cols], axis=1)
del data['participant_gender']

def convert_district(x):
    dist = str(x)
    return '0'+dist if len(dist) == 1 else dist

data.congressional_district = data.congressional_district.apply(convert_district)

desc=chars-set(incident_types)
descriptors=pd.Series(range(len(data)))
desc_li=list(desc)
for d in desc_li:
    col=data.incident_characteristics.apply(search_mask,search_chars=[d])
    descriptors=pd.concat([descriptors,col],axis=1)
descriptors.drop(0,axis=1,inplace=True)
descriptors.columns=desc_li
data = pd.concat([data, descriptors], axis=1)

# from urllib.request import Request, urlopen
# from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
# from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.support.wait import WebDriverWait
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver import ActionChains
# from selenium import webdriver
# from shapely.geometry import shape, Point
# import clipboard
# import shapefile
# import zipfile
# import shutil
# import wget
# import us

# shapefiles = '/home/toffer/GradSchool/DataMiningI/FinalProject/Shapefiles/'
# cd_filepath = shapefiles+'CongressionalDistricts/{}'
# county_filepath = shapefiles+'Counties/{}'

# states = {}
# state_counties = {}
# state_districts = {}

# def get_shapefiles(url, filename):
#     try:
#         wget.download(url, out=filename)
#         return True
#     except:
#         return False

# def get_counties(state):
#     county_filepath = shapefiles+'Counties/{}'
#     county_sf_dl = state.shapefile_urls()['county']
#     county_sf = '/tl_2010_{}_county10.shp'

#     if not os.path.exists(county_filepath.format(str(state))):
#         while not get_shapefiles(county_sf_dl, county_filepath.format(str(state))+'.zip'):
#             print('Retrying...')
#             time.sleep(0.5)
#         with zipfile.ZipFile(county_filepath.format(str(state))+'.zip', 'r') as zip_ref:
#             zip_ref.extractall(county_filepath.format(str(state)))
#         os.remove(county_filepath.format(str(state))+'.zip')
        
#     try:
#         sfc = shapefile.Reader(county_filepath.format(str(state))+county_sf.format(fips))
#         counties_di = {county.record.COUNTYFP10:county.record.NAMELSAD10 for county in sfc.shapeRecords()}
#         sfc.close()
#         return {fips:counties_di}
#     except UnicodeDecodeError:
#         sfc = shapefile.Reader(county_filepath.format(str(state))+county_sf.format(fips), encoding='latin1')
#         counties_di = {county.record.COUNTYFP10:county.record.NAMELSAD10 for county in sfc.shapeRecords()}
#         sfc.close()
#         return {fips:counties_di}
    
# def get_districts(state):
#     cd_filepath = shapefiles+'CongressionalDistricts/{}'
#     cd_sf_dl = state.shapefile_urls()['cd']
#     cd_sf = '/tl_2010_{}_cd111.shp'
    
#     if not os.path.exists(cd_filepath.format(str(state))):
#         while not get_shapefiles(cd_sf_dl, cd_filepath.format(str(state))+'.zip'):
#             print('Retrying...')
#             time.sleep(0.5)
#         with zipfile.ZipFile(cd_filepath.format(str(state))+'.zip', 'r') as zip_ref:
#             zip_ref.extractall(cd_filepath.format(str(state)))
#         os.remove(cd_filepath.format(str(state))+'.zip')
        
#     try:
#         sfd = shapefile.Reader(cd_filepath.format(str(state))+cd_sf.format(fips))
#         districts_li = [district.record.CD111FP for district in sfd.shapeRecords()]
#         sfd.close()
#         return {fips:districts_li}
#     except UnicodeDecodeError:
#         sfd = shapefile.Reader(cd_filepath.format(str(state))+cd_sf.format(fips), encoding='latin1')
#         districts_li = [district.record.CD111FP for district in sfd.shapeRecords()]
#         sfd.close()
#         return {fips:districts_li}
    
# for i in range(57):
#     fips = str(i).zfill(2)
#     state = us.states.lookup(fips)
#     if state is not None:
#         states.update({fips:str(state)})
#         state_districts.update(get_districts(state))
#         print('Finished {} districts...'.format(str(state)))
#         state_counties.update(get_counties(state))
#         print('Finished {} counties...'.format(str(state)))

# state_to_fips = {y:x for x,y in states.items()}
# cd_filepath = shapefiles+'CongressionalDistricts/{}'
# county_filepath = shapefiles+'Counties/{}'

# def get_county(x):
#     p = Point(x.longitude, x.latitude)
#     county_code = state_to_fips[x.state]
#     county_sf = '/tl_2010_{}_county10.shp'.format(county_code)
#     try:
#         sfc = shapefile.Reader(county_filepath.format(x.state)+county_sf)
#         for county in sfc.shapeRecords():
#             shp = shape(county.shape.__geo_interface__)
#             if p.within(shp):
#                 county_code += county.record.COUNTYFP10
#                 break
#         sfc.close()
#     except UnicodeDecodeError:
#         sfc = shapefile.Reader(county_filepath.format(x.state)+county_sf, encoding='latin1')
#         for county in sfc.shapeRecords():
#             shp = shape(county.shape.__geo_interface__)
#             if p.within(shp):
#                 county_code += county.record.COUNTYFP10
#                 break
#         sfc.close()
#     return county_code

# counties = data.apply(get_county, axis=1)

# profile = FirefoxProfile('/home/toffer/.mozilla/firefox/gpio495d.default')
# profile.set_preference("browser.download.panel.shown", False)
# profile.set_preference("browser.download.folderList", 2);
# profile.set_preference("browser.download.dir", "~/GradSchool/DataMiningI/FinalProject/CensusData")

# driver = webdriver.Firefox(firefox_profile=profile)

# districts_url = 'https://data.census.gov/cedsci/table?q=&y={}&g=500{}00US{}{}&table=S2301&t=Employment&tid=ACSST5Y{}.S2301&hidePreview=true'
# counties_url = 'https://data.census.gov/cedsci/table?q=&y={}&g=0500000US{}{}&table=S2301&t=Employment&tid=ACSST5Y{}.S2301&hidePreview=true'

# years = [i for i in range(2013, 2019)]
# sessions = {2013:13, 2014:14, 2015:14, 2016:15, 2017:15, 2018:16}

# def get_table(year, state, subregion, county=True):
#     base_cell = '_BOXHEAD4_EST' if year > 2016 else '_BOXHEAD#HC04_ESTIMATE#EST'
#     district_cell = 'Congressional District {} (1{}th Congress), {}'+base_cell
#     county_cell = '{}, {}'+base_cell
#     table_url = ''
#     cell = ''
#     if county:
#         table_url = counties_url.format(year, state, subregion, year)
#         county_name = state_counties[state][subregion]
#         cell = county_cell.format(county_name, states[state])
#     else:
#         table_url = districts_url.format(year, sessions[year], state, subregion, year)
#         district_number = int(subregion)
#         cell = district_cell.format(district_number, sessions[year], states[state])
#     driver.get(table_url)
#     wait = WebDriverWait(driver, 30)
#     actionChains = ActionChains(driver)
#     try:
#         wait.until(
#             EC.visibility_of_element_located(
#                 (By.CSS_SELECTOR,'[row-index="0"] > [col-id="{}"]'.format(cell))
#             )
#         )
#         src_el = driver.find_element_by_css_selector('[row-index="0"] > [col-id="{}"]'.format(cell))
        
#         dest_idx = '12' if year > 2014 else '9'
#         wait.until(
#             EC.visibility_of_element_located(
#                 (By.CSS_SELECTOR,'[row-index="{}"] > [col-id="{}"]'.format(dest_idx, cell))
#             )
#         )
#         dest_el = driver.find_element_by_css_selector('[row-index="{}"] > [col-id="{}"]'.format(dest_idx, cell))

#         actionChains.drag_and_drop(src_el, dest_el).context_click().perform()
#         wait.until(
#             EC.visibility_of_element_located(
#                 (By.CLASS_NAME, 'ag-menu-option-text')
#             )
#         )
#         dest_el = driver.find_element_by_class_name('ag-menu-option-text')
#         actionChains.move_to_element(dest_el).move_by_offset(0,5).click().perform()
#         ret_val = re.split('%\s*', clipboard.paste())[:-1]
#         ret_len = 11 if year > 2014 else 8
#         while len(ret_val) != ret_len:
#             ret_val = re.split('%\s*', clipboard.paste())[:-1]
#         return np.array(list(map(float, ret_val)))

#     except TimeoutException as e:
#         print(e)
#         return None