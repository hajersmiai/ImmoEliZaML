import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataCleaner:
##################################################################################################
#    """                                                                                         #
#   DataCleaner class for preparing real estate datasets for Machine Learning ingestion.         #
#   Includes file loading, deduplication, cleaning, type conversions, and feature normalization. #
#   """                                                                                          #
##################################################################################################
    def __init__(self, data_file_path: str, postcode_file_path: str = "data/code-postaux-belge.csv") -> None:
      
    # Initializes the DataCleaner object with data and postal code file paths.
      
        self.data_file_path = data_file_path
        self.postcode_file_path = postcode_file_path

    def load_data_file(self) -> pd.DataFrame:
##################################################################################################
#        Loads a data file in various supported formats and returns it as a DataFrame.           #
#       Supported formats: CSV, JSON, Excel, Parquet, TXT, XML.                                  #
##################################################################################################
        # Check if the file exists and is not empty
        if not os.path.exists(self.data_file_path) or os.path.getsize(self.data_file_path) == 0:
            print(f"[WARNING] File is missing or empty: {self.data_file_path}")
            return pd.DataFrame()

        suffix = Path(self.data_file_path).suffix.lower()  # Get file extension

        try:
            # Load file based on detected format
            match suffix:
                case ".csv":
                    df = pd.read_csv(self.data_file_path, low_memory=False)
                case ".json":
                    df = pd.read_json(self.data_file_path)
                case ".xls" | ".xlsx":
                    df = pd.read_excel(self.data_file_path)
                case ".parquet":
                    df = pd.read_parquet(self.data_file_path)
                case ".txt":
                    df = pd.read_csv(self.data_file_path, delimiter="\t")
                case ".xml":
                    df = pd.read_xml(self.data_file_path)
                case _:
                    print(f"[ERROR] Unsupported file format: {suffix}")
                    return pd.DataFrame()

            # Warn if the DataFrame is empty
            if df.empty:
                print(f"[WARNING] File loaded but contains no data: {self.data_file_path}")
                return pd.DataFrame()

            print(f"[INFO] Loaded {self.data_file_path} with {len(df)} rows.")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to read {self.data_file_path}: {e}")
            return pd.DataFrame()

    def analyze_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
##################################################################################################
#      Returns a summary of the DataFrame's data types, missing value counts,                    #
#      missing percentages, and unique value counts per column.                                  #
##################################################################################################
        if df.empty:
            print("[WARNING] The DataFrame is empty.")
            return pd.DataFrame()

        summary = pd.DataFrame({
            "Data type": df.dtypes,
            "Non-null count": df.notnull().sum(),
            "Missing count": df.isnull().sum(),
            "Missing %": df.isnull().mean() * 100,
            "Unique values": df.nunique()
        })

        summary = summary.sort_values(by="Missing %", ascending=False)
        return summary

    def clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
##################################################################################################
#        Removes duplicate rows and drops irrelevant columns if present.                         #
#        Displays data quality before and after cleaning.                                        #
##################################################################################################
        print("[INFO] Data Quality BEFORE cleaning:")
        summary_before = self.analyze_data_quality(df)
        print(summary_before)

        # Remove exact duplicate rows
        cleaned_df = df.drop_duplicates()

        # Drop irrelevant columns if they exist
        columns_to_drop = [
            "monthlyCost", "accessibleDisabledPeople", "hasBalcony",
            "url", "Unnamed: 0", "id"
        ]
        cleaned_df.drop(columns=[col for col in columns_to_drop if col in cleaned_df.columns], inplace=True)

        print("\n[INFO] Data Quality AFTER cleaning:")
        summary_after = self.analyze_data_quality(cleaned_df)
        print(summary_after)

        return cleaned_df

    @staticmethod
    def load_official_postcode_locality_mapping(filepath: str) -> dict:
##################################################################################################
#       Loads official Belgian postcode-locality mapping and returns it as a dictionary          #
#        for consistent locality naming.                                                         #
##################################################################################################
        # Read the CSV with explicit dtype
        df = pd.read_csv(filepath, dtype={"Code": str, "Localite": str}, sep=";")

        # Ensure required columns are present
        required_columns = {"Code", "Localite"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in {filepath}: {missing}")

        # Clean locality strings and zero-pad postcodes
        df["Localite"] = df["Localite"].str.upper().str.strip()
        df["Code"] = df["Code"].str.zfill(4)

        # Remove potential duplicates
        df = df.drop_duplicates(subset=["Code"])

        # Create and return the mapping dictionary
        mapping = dict(zip(df["Code"], df["Localite"]))
        return mapping

    def clean_errors(self) -> pd.DataFrame:
##################################################################################################
#       Performs error cleaning, including:                                                      #
#       - Locality standardization based on postal codes.                                        #
#       - Dropping irrelevant columns with excessive missing data.                               #
#       - Dropping rows missing target 'price'.                                                  #
#       - Safe type conversions for ML readiness.                                                #
##################################################################################################
        # Load and clean duplicates
        df = self.load_data_file()
        if df.empty:
            return df
        df = self.clean_duplicates(df)

        # Standardize 'locality' to uppercase
        if "locality" in df.columns:
            df["locality"] = df["locality"].astype(str).str.upper().str.strip()

        # Replace 'locality' values using official mapping
        if "postCode" in df.columns:
            official_locality = self.load_official_postcode_locality_mapping(self.postcode_file_path)
            df["locality"] = df["postCode"].map(official_locality).fillna(df["locality"])

        print("[INFO] Localities standardized using postal code mapping.")

        # Drop 'streetFacadeWidth' if present
        if "streetFacadeWidth" in df.columns:
            df.drop("streetFacadeWidth", axis=1, inplace=True)

        # Drop rows where 'price' is missing
        df = df.dropna(subset=["price"])

        # Define columns for safe type conversion
        int_cols = [
            "hasAirConditioning", "hasSwimmingPool", "hasDressingRoom", "hasFireplace",
            "hasThermicPanels", "hasArmoredDoor", "hasHeatPump", "hasPhotovoltaicPanels",
            "hasOffice", "hasAttic", "hasDiningRoom", "hasVisiophone", "hasGarden",
            "gardenSurface", "parkingCountOutdoor", "hasLift", "roomCount", "parkingCountIndoor",
            "hasBasement", "floorCount", "hasLivingRoom", "hasTerrace", "buildingConstructionYear",
            "facedeCount", "toiletCount", "bathroomCount", "bedroomCount", "postCode",
            "diningRoomSurface", "kitchenSurface", "terraceSurface", "livingRoomSurface",
            "landSurface", "habitableSurface"
        ]

        str_cols = [
            "gardenOrientation", "terraceOrientation", "kitchenType", "floodZoneType",
            "heatingType", "buildingCondition", "epcScore", "subtype", "province",
            "locality", "type"
        ]

        # Convert integer columns safely for ML (float with NaN)
        for col in int_cols:
            if col in df.columns:
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                after_na = df[col].isna().sum()
                print(f"[INFO] {col}: NaN before={before_na}, NaN after={after_na}")

        # Convert string columns with 'unknown' for missing
        for col in str_cols:
            if col in df.columns:
                before_na = df[col].isna().sum()
                df[col] = df[col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
                after_na = df[col].isna().sum()
                print(f"[INFO] {col}: missing before={before_na}, missing after={after_na}")

        print("[INFO] Data cleaned and prepared for ML ingestion.")
        return df

    def normalization(self) -> pd.DataFrame:
##################################################################################################
#       Normalizes specific categorical columns to numerical values suitable for ML models.      #
##################################################################################################
        df = self.clean_errors()

        # Define normalization mappings for categorical features
        mappings = {
            "buildingCondition": {
                "missing value": np.nan, "GOOD": 1, "AS_NEW": 2, "TO_RENOVATE": 3,
                "TO_BE_DONE_UP": 4, "JUST_RENOVATED": 5, "TO_RESTORE": 6
            },
            "epcScore": {
                "missing value": np.nan, 'A++': 1, 'A+': 2, 'A': 3, 'B': 4, 'C': 5,
                'D': 6, 'E': 7, 'F': 8, 'G': 9, 'G_C': 9, 'F_D': 8, 'C_A': 5,
                'F_C': 8, 'E_C': 7, 'C_B': 5, 'E_D': 7, 'G_F': 9, 'D_C': 6, 'G_E': 9,
                'X': np.nan
            },
            "heatingType": {
                "missing value": np.nan, 'GAS': 1, 'FUELOIL': 2, 'ELECTRIC': 3,
                'PELLET': 4, 'WOOD': 5, 'SOLAR': 6, 'CARBON': 7
            },
            "floodZoneType": {
                "missing value": np.nan, 'NON_FLOOD_ZONE': 1, 'POSSIBLE_FLOOD_ZONE': 2,
                'RECOGNIZED_FLOOD_ZONE': 3, 'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 4,
                'CIRCUMSCRIBED_WATERSIDE_ZONE': 5, 'CIRCUMSCRIBED_FLOOD_ZONE': 6,
                'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 7, 'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 8,
                'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 9
            },
            "kitchenType": {
                "missing value": np.nan, 'NOT_INSTALLED': 0, 'SEMI_EQUIPPED': 1,
                'INSTALLED': 2, 'HYPER_EQUIPPED': 3, 'USA_UNINSTALLED': 0,
                'USA_SEMI_EQUIPPED': 1, 'USA_INSTALLED': 2, 'USA_HYPER_EQUIPPED': 3
            }
        }

        # Apply each mapping to create encoded columns for ML
        for col, mapping in mappings.items():
            if col in df.columns:
                new_col = f"{col}_enc"
                # Replace using the mapping
                df[new_col] = df[col].replace(mapping)

                # Replace 'unknown' and other non-mapped strings with np.nan
                df[new_col] = df[new_col].replace("unknown", np.nan)

                # Finally convert to float
                df[new_col] = df[new_col].astype(float)
                print(f"[INFO] Encoded '{col}' → '{new_col}' with {df[new_col].isna().sum()} NaNs ready for ML.")

        return df

    def to_real_values(self) -> pd.DataFrame:
##################################################################################################
#        Final step: returns a fully cleaned, normalized DataFrame ready for ML pipelines.       #
##################################################################################################
        df = self.normalization()
        print("[INFO] DataFrame fully prepared for ML model ingestion.")
        return df

    def send_output_file(self, output_file: str) -> None:
        """
        Exports the cleaned and normalized DataFrame to a CSV file.
        """
        df = self.to_real_values()
        if not df.empty:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"[SUCCESS] Exported {len(df)} cleaned records to {output_file}")
        else:
            print("[WARNING] Export aborted: DataFrame is empty.")
