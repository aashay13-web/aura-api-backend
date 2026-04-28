from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import fitz                                    
from PIL import Image            
import google.generativeai as genai
from google.generativeai import GenerativeModel  # type: ignore
import json
import os
from dotenv import load_dotenv

# Load secret environment variables from the .env file
load_dotenv()

app = FastAPI(title="Aura PropTech API")

# Securely configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore

# Allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DYNAMIC_VALUATION_DB = {}

from typing import List, Optional

# --- DATA MODELS ---
class PlotQuery(BaseModel):
    region_code: str
    zoning: str
    road_access: str
    area: float
    area_unit: str  # "kattha", "dismil", or "sqft"
    rate_unit: str  # "kattha", "dismil", or "sqft"

class PersonInfo(BaseModel):
    name: str
    age: str = ""
    caste: str = ""
    identity: str = ""
    address: str = ""

class Boundaries(BaseModel):
    north: str = ""
    south: str = ""
    east: str = ""
    west: str = ""

class DeedRequest(BaseModel):
    language: str = "hindi"  # The frontend will send "hindi" or "english" here
    property_type: str = "Sale Deed"
    registration_date: str = ""
    property_location: str
    thana_details: str = ""
    khata_number: str = ""
    plot_khesra_number: str = ""
    plot_size: str
    property_valuation: str
    zoning_hint: str = ""
    boundaries_chauhaddi: Boundaries
    buyers: list[PersonInfo]
    sellers: list[PersonInfo]
    identifiers: list[PersonInfo] = []
    witnesses: list[PersonInfo] = []
    special_clauses: str = "Standard absolute transfer of ownership."

@app.get("/")
async def root():
    return {"message": "Aura AI Engine is live."}

# --- ENDPOINT 1: Upload Valuation Charts ---
@app.post("/api/upload-valuations")
async def upload_valuations(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV or Excel files accepted.")

    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        df['Zoning_Type'] = df['Zoning_Type'].astype(str).str.lower().str.strip()
        df['Road_Access'] = df['Road_Access'].astype(str).str.lower().str.strip()
        
        global DYNAMIC_VALUATION_DB
        DYNAMIC_VALUATION_DB = {} 
        
        for index, row in df.iterrows():
            region = str(row['Region_Code'])
            zoning = row['Zoning_Type']
            road = row['Road_Access']
            rate = row['Rate_Per_Unit']
            
            if region not in DYNAMIC_VALUATION_DB:
                DYNAMIC_VALUATION_DB[region] = {}
            if zoning not in DYNAMIC_VALUATION_DB[region]:
                DYNAMIC_VALUATION_DB[region][zoning] = {}
                
            DYNAMIC_VALUATION_DB[region][zoning][road] = rate

        return {"status": "success", "message": f"Ingested {len(df)} valuation records."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- ENDPOINT 2: Calculate Plot Valuation ---
@app.post("/api/calculate-valuation")
async def calculate_valuation(query: PlotQuery):
    region = query.region_code
    zoning = query.zoning.lower().strip()
    road = query.road_access.lower().strip()
    
    if region not in DYNAMIC_VALUATION_DB:
        raise HTTPException(status_code=404, detail=f"No data found for region: {region}")
        
    try:
        base_rate = DYNAMIC_VALUATION_DB[region][zoning][road]
    except KeyError:
        raise HTTPException(status_code=404, detail="No matching rate found for this specific zoning and road access.")
        
    input_unit = query.area_unit.lower()
    if input_unit == "kattha":
        area_sqft = query.area * 1361.25
    elif input_unit == "dismil":
        area_sqft = query.area * 435.6
    elif input_unit == "sqft":
        area_sqft = query.area
    else:
        raise HTTPException(status_code=400, detail="Invalid area_unit. Use kattha, dismil, or sqft.")

    area_kattha = area_sqft / 1361.25
    area_dismil = area_sqft / 435.6

    rate_unit = query.rate_unit.lower()
    if rate_unit == "kattha":
        total_value = base_rate * area_kattha
    elif rate_unit == "dismil":
        total_value = base_rate * area_dismil
    elif rate_unit == "sqft":
        total_value = base_rate * area_sqft
    else:
        raise HTTPException(status_code=400, detail="Invalid rate_unit. Use kattha, dismil, or sqft.")
        
    return {
        "status": "success",
        "plot_details": {"region": region, "zoning": zoning, "road_access": road},
        "measurements": {"sqft": round(area_sqft, 2), "dismil": round(area_dismil, 3), "kattha": round(area_kattha, 3)},
        "valuation": {"rate_applied": f"₹{base_rate} per {rate_unit}", "total_estimated_value": round(total_value, 2)}
    }

# --- ENDPOINT 3: Deep Document Analyzer (VISION UPGRADE) ---
@app.post("/api/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Please upload a PDF document.")
        
    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        
        # 1. Convert PDF pages directly to Images
        images_for_gemini = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images_for_gemini.append(img)
                
        # 2. Tell the AI to look at the layout carefully
        llm_prompt = """
        You are an expert Indian Real Estate legal assistant. 
        I am providing you with images of a scanned property deed. It contains a mix of English and Hindi.
        Read the document natively and extract the information into a valid JSON object.
        Pay extreme attention to the visual layout so you do not mix up Buyer and Seller details. 
        Look carefully for:
        - "W/O" or "S/O" or "D/O" (Wife/Son/Daughter of) for identities.
        - Caste or Community (जाति) often mentioned right after the name or age.
        - The exact unit of measurement (Dismil, Sqft, Kattha). Do not guess.
        - Names of people in the Chauhaddi (Boundaries) section.
        - Correct Thana numbers and PAN details linked to the specific person.
        - CRITICAL: Map "क्रेता" (Kreta), "Claimant", or "Purchaser" ONLY to the 'buyers' array.
        - CRITICAL: Map "विक्रेता" (Vikreta), "Executant", or "Vendor" ONLY to the 'sellers' array.
        - CRITICAL: Distinctly separate the Identifier (पहचानकर्ता / Pehchaan) from the Witnesses (गवाह / Gawah).

        Required JSON Output format:
        {
            "property_summary": "A 2-3 sentence clear summary of the transaction.",
            "property_type": "Specify if Khatiani, Kewala, Ancestral, etc.",
            "previous_owners": ["name1", "name2"],
            "registration_date": "Extracted date of registry",
            "property_location": "Detailed address, village, or district",
            "thana_details": "Thana name and Thana number",
            "khata_number": "Khata number (खाता नंबर)",
            "plot_khesra_number": "Plot or Khesra number (खेसरा नंबर)",
            "mutation_or_rent_demand": "Details of any Lagan or mutation demand",
            "plot_size": "Total extracted size WITH exact unit (e.g., 6.30 Dismil)",
            "property_valuation": "Sale price or exact valuation",
            "zoning_hint": "Agricultural/Commercial/Residential",
            "boundaries_chauhaddi": {
                "north": "...",
                "south": "...",
                "east": "...",
                "west": "..."
            },
            "buyers": [
                {
                    "name": "...",
                    "age": "...",
                    "caste": "...",
                    "identity": "Aadhar, PAN, or relative's name (e.g., S/O, W/O)",
                    "address": "..."
                }
            ],
            "sellers": [
                {
                    "name": "...",
                    "age": "...",
                    "caste": "...",
                    "identity": "Aadhar, PAN, or relative's name (e.g., S/O, W/O)",
                    "address": "..."
                }
            ],
            "identifiers": [
                {
                    "name": "...",
                    "age": "...",
                    "caste": "...",
                    "identity": "Relative's name (e.g., S/O, W/O), Aadhar, PAN, etc.",
                    "address": "..."
                }
            ],
            "witnesses": [
                {
                    "name": "...",
                    "age": "...",
                    "caste": "...",
                    "identity": "Relative's name (e.g., S/O, W/O), Aadhar, PAN, etc.",
                    "address": "..."
                }
            ]
        }
        """
        # 3. Send the Prompt AND the Images to Gemini
        model = GenerativeModel('gemini-2.5-flash')
        
        # Combine the prompt and the images into one request
        request_content = [llm_prompt] + images_for_gemini
        response = model.generate_content(request_content)
        
        try:
            clean_response = response.text.replace("```json", "").replace("```", "").strip()
            extracted_data = json.loads(clean_response)
        except json.JSONDecodeError:
            extracted_data = {"error": "Failed to parse AI output", "raw_text": response.text}

        return {
            "status": "success",
            "filename": file.filename,
            "extraction_method": "Gemini Native Vision (Bypassing Tesseract)",
            "extracted_data": extracted_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    
# --- ENDPOINT 4: Pure Bilingual 9-Page Formatted Deed Writer ---
@app.post("/api/generate-deed")
async def generate_deed(request: DeedRequest):
    try:
        structured_data = request.model_dump_json(indent=2)
        
        if request.language.lower() == "english":
            # 100% PURE ENGLISH TEMPLATE
            drafting_prompt = f"""
            Act as an expert legal draftsman in Bihar, India.
            Draft a formal 'Sale Deed' using this JSON data: {structured_data}

            CRITICAL REQUIREMENT: Write the ENTIRE document completely in professional Indian Legal English. No Hindi words should be used. Use this exact 9-page layout:

            ### --- PAGE 1 ---
            **INDIA NON JUDICIAL**
            **Rs. 500 / FIVE HUNDRED RUPEES**
            **Summary of Endorsement**
            [State registration date and Seller's name presenting the document. Add execution acceptance text.]
            **SALE DEED**
            [List Seller's name, total plot size, and actual valuation]

            ### --- PAGE 2 ---
            **Sub District Registry Office, [Extract Thana Details]**
            **(Biometrics, Photographs, and Fingerprint Index)**
            * **Claimant (Buyer):** [Buyer Name] | *[Photo]* | *[Thumb Index]* | *[Signature]*
            * **Presented By (Executant):** [Seller Name] | *[Photo]* | *[Thumb Index]* | *[Signature]*
            * **Identified By:** [Identifier Name] | *[Photo]* | *[Thumb Index]* | *[Signature]*

            ### --- PAGE 3 ---
            **1. Details of Executant (Seller):**
            [Write paragraph with Seller's Name, Age, Father/Husband, Caste, Address, Aadhar/PAN]
            **2. Details of Claimant (Buyer):**
            [Write paragraph with Buyer's Name, Age, Father/Husband, Caste, Address, Aadhar/PAN]
            **3. Nature of Document:** [Property Type]
            **4. Valuation:**
            [State Government Valuation and Actual Sale Price]

            ### --- PAGE 4 ---
            **5. Schedule of Property:**
            [Detail plot size, location, Thana, District]
            * **Khata No.:** [Khata]
            * **Plot/Khesra No.:** [Khesra]
            * **Annual Rent (Lagan):** [Lagan/Demand]
            **Boundaries:**
            * **North:** [North Boundary]
            * **South:** [South Boundary]
            * **East:** [East Boundary]
            * **West:** [West Boundary]

            ### --- PAGE 5 ---
            **Terms & Conditions:**
            [Draft Clauses 1, 2, 3 detailing the need for money, agreed price, and transfer of absolute ownership.]

            ### --- PAGE 6 ---
            [Draft Clauses 4, 5 detailing future mutation rights, guarantee of no encumbrances, and holding seller liable for future title defects.]

            ### --- PAGE 7 ---
            [State the document was signed in sound mind. Create signature lines for Seller, Buyer, and Witnesses with details.]

            ### --- PAGE 8 ---
            **Map of Plot (Nazri Naksha)**
            [State Location, Khata, Khesra, Area]
            > *[Graphical representation of Plot [Khesra] drawn here]*
            > **North:** [North Boundary]
            > **South:** [South Boundary]
            > **East:** [East Boundary]
            > **West:** [West Boundary]
            [Property shown in red ink.]

            ### --- PAGE 9 ---
            **Endorsements & Admissibility**
            [Standard English text about Indian Stamp Act, presentation, execution admission, and registration completion.]
            
            Do NOT use markdown code blocks. Return raw text.
            """

        else:
            # 100% PURE HINDI TEMPLATE
            drafting_prompt = f"""
            Act as an expert legal draftsman in Bihar, India.
            Draft a formal 'Sale Deed' using this JSON data: {structured_data}

            CRITICAL REQUIREMENT: Write the ENTIRE document completely in pure Hindi (Devanagari script), exactly as standard Bihar registries use. No English words should be used except where absolutely necessary for IDs. Use this exact 9-page layout:

            ### --- PAGE 1 ---
            **भारतीय गैर न्यायिक**
            **500 रुपये / पाँच सौ रुपये**
            **पृष्ठांकन का सारांश**
            [State registration date and Seller's name presenting the document. Add execution acceptance text in Hindi.]
            **विक्रय पत्र**
            [List Seller's name, total plot size, and actual valuation in Hindi]

            ### --- PAGE 2 ---
            **अवर निबंधन कार्यालय, [Extract Thana Details]**
            **(बायोमेट्रिक्स, फोटो और फिंगरप्रिंट)**
            * **लेख्यधारी (क्रेता):** [Buyer Name] | *[फोटो]* | *[अंगूठे का निशान]* | *[हस्ताक्षर]*
            * **लेख्यकारी (विक्रेता):** [Seller Name] | *[फोटो]* | *[अंगूठे का निशान]* | *[हस्ताक्षर]*
            * **पहचानकर्ता:** [Identifier Name] | *[फोटो]* | *[अंगूठे का निशान]* | *[हस्ताक्षर]*

            ### --- PAGE 3 ---
            **1. लेख्यकारी (विक्रेता) का विवरण:**
            [Write paragraph with Seller's Name, Age, Father/Husband, Caste, Address, Aadhar/PAN in Hindi]
            **2. लेख्यधारी (क्रेता) का विवरण:**
            [Write paragraph with Buyer's Name, Age, Father/Husband, Caste, Address, Aadhar/PAN in Hindi]
            **3. लेख्य-प्रकार:** [Property Type]
            **4. विक्रय-मूल्य:**
            [State Government Valuation and Actual Sale Price in Hindi]

            ### --- PAGE 4 ---
            **5. विक्रय-सम्पति का विवरण:**
            [Detail plot size, location, Thana, District in Hindi]
            * **खाता सं०:** [Khata]
            * **प्लौट सं० (खेसरा):** [Khesra]
            * **लगान:** [Lagan/Demand]
            **चौहदी:**
            * **उतर:** [North Boundary]
            * **दक्षिण:** [South Boundary]
            * **पुरब:** [East Boundary]
            * **पश्चिम:** [West Boundary]

            ### --- PAGE 5 ---
            **शर्तें:**
            [Draft Clauses 1, 2, 3 detailing the need for money, agreed price, and transfer of absolute ownership in Hindi.]

            ### --- PAGE 6 ---
            [Draft Clauses 4, 5 detailing future mutation rights, guarantee of no encumbrances, and holding seller liable for future title defects in Hindi.]

            ### --- PAGE 7 ---
            [State the document was signed in sound mind. Create signature lines for Seller, Buyer, and Witnesses with details in Hindi.]

            ### --- PAGE 8 ---
            **नजरी नक्शा**
            [State Location, Khata, Khesra, Area]
            > *[प्लौट [Khesra] का नक्शा]*
            > **उतर:** [North Boundary]
            > **दक्षिण:** [South Boundary]
            > **पुरब:** [East Boundary]
            > **पश्चिम:** [West Boundary]
            [विक्रय-सम्पदा को लाल रंग से दर्शाया गया है।]

            ### --- PAGE 9 ---
            **प्रमाणपत्र एवं पृष्ठांकन**
            [Standard Hindi text about Indian Stamp Act, presentation, execution admission, and registration completion.]
            
            Do NOT use markdown code blocks. Return raw text.
            """
        
        model = GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(drafting_prompt)
        
        return {
            "status": "success",
            "message": f"9-Page Formatted Deed generated successfully in {request.language.upper()}.",
            "document_draft": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate deed: {str(e)}")