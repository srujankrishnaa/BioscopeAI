# Inference and Heatmap Generation for Above Ground Biomass Prediction
# Enhanced version with better efficiency, visualization, and reporting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import logging
import contextlib
import io
import sys
from PIL import Image
import requests
from shapely.geometry import box, Point
import geopandas as gpd
import folium
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import json

# Import the data processor from preprocessing
from data_preprocessing import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 4.1 Enhanced Inference Pipeline
class BiomassPredictor:
    def __init__(self, model_path, scalers_path, common_coords_path, config=None):
        """
        Initialize the biomass predictor with enhanced error handling and validation
        
        Args:
            model_path: Path to the quantized TensorFlow Lite model
            scalers_path: Path to the saved preprocessing scalers
            common_coords_path: List of paths to common longitude and latitude arrays
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model_path = model_path
        self.scalers_path = scalers_path
        self.common_coords_path = common_coords_path
        self.use_tflite = False
        self.interpreter = None
        self.model = None
        self.scalers = {}
        self.common_lons = None
        self.common_lats = None
        self.processor = None
        self.data_cache = {}
        self.city_boundaries = None
        
        # Initialize components with validation
        self._load_model()
        self._load_preprocessing_params()
        self._load_coordinates()
        self._initialize_processor()
        self._load_city_boundaries()
        self._create_output_directories()
        
        # Initialize GLM-4.5 API if available
        self.glm_api_key = self.config.get('glm_api_key')
        self.use_glm = self.glm_api_key is not None
        
        logger.info("BiomassPredictor initialized successfully")

    def _load_model(self):
        """Enhanced model loading with input shape validation"""
        try:
            # Try loading TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Log model details
            input_shape = self.input_details[0]['shape']
            output_shape = self.output_details[0]['shape']
            logger.info(f"Loaded TFLite model from: {self.model_path}")
            logger.info(f"Input shape: {input_shape}, Output shape: {output_shape}")
            
            self.use_tflite = True
            self.expected_input_shape = input_shape
            
        except Exception as e:
            logger.warning(f"Could not load TFLite model: {e}")
            try:
                # Try multiple fallback paths
                fallback_paths = [
                    self.model_path.replace('.tflite', '.h5'),
                    './models/biomass_cnn_lstm_best.h5',
                    './models/biomass_cnn_lstm.h5'
                ]
                
                for path in fallback_paths:
                    if os.path.exists(path):
                        self.model = tf.keras.models.load_model(path)
                        self.use_tflite = False
                        logger.info(f"Loaded Keras model from: {path}")
                        
                        # Get model summary
                        self.model.summary()
                        break
                else:
                    raise RuntimeError("No valid model found")
            except Exception as e2:
                logger.error(f"Could not load any model: {e2}")
                raise

    def _load_city_boundaries(self):
        """Load city boundaries for better visualization"""
        try:
            # Try to load city boundaries from GeoJSON
            boundaries_path = self.config.get('city_boundaries_path', './data/city_boundaries.geojson')
            if os.path.exists(boundaries_path):
                self.city_boundaries = gpd.read_file(boundaries_path)
                logger.info(f"Loaded city boundaries from {boundaries_path}")
            else:
                logger.warning("City boundaries file not found. Using simple bounding boxes.")
                self.city_boundaries = None
        except Exception as e:
            logger.warning(f"Could not load city boundaries: {e}")
            self.city_boundaries = None

    def _create_output_directories(self):
        """Create organized output directories"""
        self.output_dirs = {
            'base': './outputs',
            'heatmaps': './outputs/heatmaps',
            'reports': './outputs/reports',
            'interactive': './outputs/interactive',
            'pdf': './outputs/pdf'
        }
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info("Output directories created")

    def predict_for_city(self, city_name, city_bbox=None, start_date="2023-01-01", end_date="2023-12-31"):
        """
        Enhanced prediction with caching and multiple output formats
        
        Args:
            city_name: Name of the city
            city_bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
        
        Returns:
            Dictionary containing all output paths and data
        """
        try:
            logger.info(f"Starting prediction for {city_name}")
            
            # Check cache first
            cache_key = f"{city_name}_{start_date}_{end_date}"
            if cache_key in self.data_cache:
                logger.info(f"Using cached data for {city_name}")
                return self.data_cache[cache_key]
            
            # Get bounding box if not provided
            if city_bbox is None:
                city_bbox = self._get_city_bbox(city_name)
                if city_bbox is None:
                    raise ValueError(f"Could not determine bounding box for {city_name}")
            
            # 1. Fetch data for the city
            logger.info("Fetching data...")
            if self.processor is None:
                raise RuntimeError("Data processor not initialized")
            
            local_datasets = self.processor.load_local_datasets()
            filtered_datasets = self._filter_to_bbox(local_datasets, city_bbox)
            
            # Fetch additional NASA data
            try:
                nasa_datasets = self.processor.fetch_nasa_data(city_bbox, start_date, end_date)
                all_datasets = {**filtered_datasets, **nasa_datasets}
            except Exception as e:
                logger.warning(f"Could not fetch NASA data: {e}")
                all_datasets = filtered_datasets
            
            # 2. Preprocess data
            logger.info("Preprocessing data...")
            qc_datasets = self.processor.apply_quality_control(all_datasets)
            aligned_datasets, _, _ = self.processor.align_datasets(qc_datasets)
            
            # Use existing scalers if available, otherwise create new ones
            if self.scalers:
                normalized_datasets = self._apply_existing_scalers(aligned_datasets)
            else:
                normalized_datasets = self.processor.normalize_data(aligned_datasets)
            
            # 3. Create sequence
            logger.info("Creating sequence...")
            sequences, _ = self.processor.create_sequences(normalized_datasets, seq_length=12)
            
            if len(sequences) == 0:
                raise ValueError("No sequences could be created from the data")
            
            # 4. Make prediction
            logger.info("Making prediction...")
            prediction = self._predict(sequences[-1:])  # Use the most recent sequence
            
            # 5. Generate visualizations
            logger.info("Generating visualizations...")
            heatmap_path = self._generate_heatmap(prediction, city_name, city_bbox)
            interactive_map_path = self._generate_interactive_map(prediction, city_name, city_bbox)
            
            # 6. Generate reports
            logger.info("Generating reports...")
            report_text = self._generate_report(city_name, prediction, city_bbox)
            pdf_report_path = self._generate_pdf_report(city_name, prediction, city_bbox, report_text)
            
            # 7. Enhance report with GLM-4.5 if available
            if self.use_glm:
                logger.info("Enhancing report with GLM-4.5...")
                enhanced_report = self._enhance_report_with_glm(city_name, prediction, city_bbox, report_text)
            else:
                enhanced_report = report_text
            
            # Compile results
            results = {
                'city_name': city_name,
                'city_bbox': city_bbox,
                'prediction': prediction,
                'heatmap_path': heatmap_path,
                'interactive_map_path': interactive_map_path,
                'report_text': enhanced_report,
                'pdf_report_path': pdf_report_path,
                'statistics': self._calculate_statistics(prediction)
            }
            
            # Cache results
            self.data_cache[cache_key] = results
            
            logger.info(f"Prediction completed for {city_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline for {city_name}: {e}")
            # Return default outputs
            return self._generate_default_results(city_name)

    def _get_city_bbox(self, city_name):
        """Get bounding box for a city using Nominatim API"""
        try:
            # Try to get from local boundaries first
            if self.city_boundaries is not None:
                city_data = self.city_boundaries[self.city_boundaries['name'] == city_name]
                if not city_data.empty:
                    bounds = city_data.geometry.bounds.iloc[0]
                    return [bounds.minx, bounds.miny, bounds.maxx, bounds.maxy]
            
            # Fallback to Nominatim API
            url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name},India&limit=1"
            response = requests.get(url, headers={'User-Agent': 'BiomassPrediction/1.0'})
            if response.status_code == 200:
                data = response.json()
                if data:
                    bbox = data[0]['boundingbox']
                    return [float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])]
            
            return None
        except Exception as e:
            logger.error(f"Error getting bbox for {city_name}: {e}")
            return None

    def _generate_interactive_map(self, prediction, city_name, city_bbox):
        """Generate interactive map with Folium"""
        try:
            # Denormalize prediction if scaler is available
            if 'gedi' in self.scalers:
                scaler = self.scalers['gedi']
                denormalized = scaler.inverse_transform(prediction.reshape(-1, 1))
                heatmap_2d = denormalized.reshape(prediction.shape[1], prediction.shape[2])
            else:
                heatmap_2d = prediction[0, :, :, 0]
            
            # Create base map centered on city
            center_lat = (city_bbox[1] + city_bbox[3]) / 2
            center_lon = (city_bbox[0] + city_bbox[2]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            
            # Create heatmap data
            heat_data = []
            lat_step = (city_bbox[3] - city_bbox[1]) / heatmap_2d.shape[0]
            lon_step = (city_bbox[2] - city_bbox[0]) / heatmap_2d.shape[1]
            
            for i in range(heatmap_2d.shape[0]):
                for j in range(heatmap_2d.shape[1]):
                    lat = city_bbox[1] + i * lat_step
                    lon = city_bbox[0] + j * lon_step
                    heat_data.append([lat, lon, heatmap_2d[i, j]])
            
            # Add heatmap layer
            from folium.plugins import HeatMap
            HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
            
            # Add city boundary
            if self.city_boundaries is not None:
                city_data = self.city_boundaries[self.city_boundaries['name'] == city_name]
                if not city_data.empty:
                    folium.GeoJson(
                        city_data.geometry.to_json(),
                        style_function=lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'fillOpacity': 0.2, 'weight': 2}
                    ).add_to(m)
            
            # Add title
            title_html = f'''
                <h3 align="center" style="font-size:16px"><b>Biomass Prediction: {city_name}</b></h3>
                <p align="center" style="font-size:12px">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Save map
            map_path = f'./outputs/interactive/{city_name}_biomass_map.html'
            m.save(map_path)
            
            logger.info(f"Interactive map saved to: {map_path}")
            return map_path
            
        except Exception as e:
            logger.error(f"Error generating interactive map: {e}")
            return None

    def _generate_pdf_report(self, city_name, prediction, city_bbox, report_text):
        """Generate PDF report with ReportLab"""
        try:
            # Create PDF document
            pdf_path = f'./outputs/pdf/{city_name}_biomass_report.pdf'
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph(f"Biomass Analysis Report: {city_name}", title_style))
            story.append(Spacer(1, 12))
            
            # Date
            date_style = ParagraphStyle(
                'CustomDate',
                parent=styles['Normal'],
                fontSize=10,
                alignment=1
            )
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", date_style))
            story.append(Spacer(1, 12))
            
            # Add heatmap
            heatmap_path = f'./outputs/heatmaps/{city_name}_biomass_heatmap.png'
            if os.path.exists(heatmap_path):
                story.append(Paragraph("Biomass Heatmap", styles['Heading2']))
                story.append(Spacer(1, 12))
                story.append(RLImage(heatmap_path, width=6*inch, height=4.5*inch))
                story.append(Spacer(1, 12))
            
            # Statistics table
            stats = self._calculate_statistics(prediction)
            story.append(Paragraph("Biomass Statistics", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            data = [
                ['Metric', 'Value'],
                ['Mean Biomass (Mg/ha)', f"{stats['mean']:.2f}"],
                ['Max Biomass (Mg/ha)', f"{stats['max']:.2f}"],
                ['Min Biomass (Mg/ha)', f"{stats['min']:.2f}"],
                ['Std Deviation', f"{stats['std']:.2f}"],
                ['Area (km²)', f"{stats['area']:.1f}"],
                ['Total Biomass (Mg)', f"{stats['total']:.0f}"]
            ]
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 12))
            
            # Report text
            story.append(Paragraph("Analysis", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            # Split report text into paragraphs
            for paragraph in report_text.split('\n\n'):
                if paragraph.strip():
                    story.append(Paragraph(paragraph, styles['Normal']))
                    story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report saved to: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None

    def _enhance_report_with_glm(self, city_name, prediction, city_bbox, base_report):
        """Enhance report using GLM-4.5 API"""
        try:
            # Calculate statistics for context
            stats = self._calculate_statistics(prediction)
            
            # Prepare prompt for GLM
            prompt = f"""
            I have generated a biomass prediction report for {city_name}. Here are the key statistics:
            - Mean Biomass: {stats['mean']:.2f} Mg/ha
            - Max Biomass: {stats['max']:.2f} Mg/ha
            - Min Biomass: {stats['min']:.2f} Mg/ha
            - Area: {stats['area']:.1f} km²
            - Total Biomass: {stats['total']:.0f} Mg
            
            Here's the current analysis:
            {base_report}
            
            Please enhance this report by:
            1. Adding insights about the environmental implications of these biomass levels
            2. Discussing potential impacts of climate change on biomass in this region
            3. Suggesting specific conservation or restoration strategies based on the biomass distribution
            4. Adding context about how this compares to national or global averages
            
            Please maintain the existing structure but enhance the content with your analysis.
            """
            
            # Call GLM-4.5 API
            headers = {
                'Authorization': f'Bearer {self.glm_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'glm-4',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }
            
            response = requests.post(
                'https://open.bigmodel.cn/api/paas/v4/chat/completions',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_content = result['choices'][0]['message']['content']
                
                # Combine base report with enhanced content
                enhanced_report = f"""
                {base_report}
                
                === ENHANCED ANALYSIS ===
                {enhanced_content}
                """
                
                return enhanced_report
            else:
                logger.warning(f"GLM API returned status code {response.status_code}")
                return base_report
                
        except Exception as e:
            logger.error(f"Error enhancing report with GLM: {e}")
            return base_report

    def _calculate_statistics(self, prediction):
        """Calculate comprehensive statistics from prediction"""
        try:
            # Denormalize prediction if scaler is available
            if 'gedi' in self.scalers:
                scaler = self.scalers['gedi']
                denormalized = scaler.inverse_transform(prediction.reshape(-1, 1))
                values = denormalized.flatten()
            else:
                values = prediction.flatten()
            
            # Calculate basic statistics
            mean_val = np.mean(values)
            max_val = np.max(values)
            min_val = np.min(values)
            std_val = np.std(values)
            
            # Calculate total biomass (simplified)
            # This would require actual area calculation in practice
            total_biomass = mean_val * 100  # Simplified calculation
            
            return {
                'mean': mean_val,
                'max': max_val,
                'min': min_val,
                'std': std_val,
                'total': total_biomass,
                'area': 100.0  # Simplified
            }
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {
                'mean': 0,
                'max': 0,
                'min': 0,
                'std': 0,
                'total': 0,
                'area': 0
            }

    def _generate_default_results(self, city_name):
        """Generate default results when prediction fails"""
        return {
            'city_name': city_name,
            'error': 'Prediction failed',
            'heatmap_path': self._generate_default_heatmap(city_name),
            'interactive_map_path': None,
            'report_text': f"Error generating prediction for {city_name}",
            'pdf_report_path': None,
            'statistics': {
                'mean': 0,
                'max': 0,
                'min': 0,
                'std': 0,
                'total': 0,
                'area': 0
            }
        }

    # Keep existing methods: _filter_to_bbox, _apply_existing_scalers, _predict, 
    # _generate_heatmap, _generate_default_heatmap, _generate_report, _calculate_bbox_area

# 4.2 Enhanced Inference Execution
def main():
    """Main inference execution function with enhanced features"""
    try:
        # Load configuration
        config_path = './config/inference_config.json'
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Initialize predictor
        logger.info("Initializing biomass predictor...")
        predictor = BiomassPredictor(
            model_path=config.get('model_path', './models/biomass_cnn_lstm_quantized.tflite'),
            scalers_path=config.get('scalers_path', './data/processed/scalers.pkl'),
            common_coords_path=config.get('common_coords_path', 
                                      ['./data/processed/common_lons.npy', './data/processed/common_lats.npy']),
            config=config
        )
        
        # Define cities to process
        cities = config.get('cities', {
            'Mumbai': [72.7, 18.9, 73.0, 19.2],
            'Delhi': [77.0, 28.4, 77.3, 28.7],
            'Bangalore': [77.4, 12.8, 77.8, 13.1],
            'Chennai': [80.1, 12.8, 80.3, 13.2]
        })
        
        # Date range for analysis
        start_date = config.get('start_date', "2023-01-01")
        end_date = config.get('end_date', "2023-12-31")
        
        # Process each city
        results = {}
        for city_name, bbox in cities.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {city_name}...")
            logger.info(f"{'='*50}")
            
            # Make prediction
            city_result = predictor.predict_for_city(
                city_name, bbox, start_date, end_date
            )
            
            results[city_name] = city_result
            
            logger.info(f"\nResults for {city_name}:")
            if 'error' not in city_result:
                logger.info(f"Heatmap: {city_result['heatmap_path']}")
                logger.info(f"Interactive Map: {city_result['interactive_map_path']}")
                logger.info(f"PDF Report: {city_result['pdf_report_path']}")
            else:
                logger.warning(f"Error: {city_result['error']}")
            
            logger.info(f"\n{'='*50}")
        
        # Generate summary report
        logger.info("Generating summary report...")
        summary_path = './outputs/reports/summary_report.html'
        generate_summary_report(results, summary_path)
        
        logger.info(f"Summary report saved to: {summary_path}")
        logger.info("\nInference pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

def generate_summary_report(results, output_path):
    """Generate HTML summary report of all processed cities"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Biomass Prediction Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                .city-card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
                .stats { display: flex; flex-wrap: wrap; }
                .stat { margin-right: 20px; margin-bottom: 10px; }
                .heatmap { max-width: 300px; height: 200px; object-fit: cover; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Biomass Prediction Summary Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        """
        
        # Summary table
        html_content += """
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>City</th>
                    <th>Mean Biomass (Mg/ha)</th>
                    <th>Max Biomass (Mg/ha)</th>
                    <th>Min Biomass (Mg/ha)</th>
                    <th>Total Biomass (Mg)</th>
                </tr>
        """
        
        for city_name, result in results.items():
            if 'error' not in result:
                stats = result['statistics']
                html_content += f"""
                    <tr>
                        <td>{city_name}</td>
                        <td>{stats['mean']:.2f}</td>
                        <td>{stats['max']:.2f}</td>
                        <td>{stats['min']:.2f}</td>
                        <td>{stats['total']:.0f}</td>
                    </tr>
                """
        
        html_content += "</table>"
        
        # City details
        html_content += "<h2>City Details</h2>"
        
        for city_name, result in results.items():
            html_content += f"""
                <div class="city-card">
                    <h3>{city_name}</h3>
                    <div class="stats">
            """
            
            if 'error' not in result:
                stats = result['statistics']
                html_content += f"""
                    <div class="stat"><strong>Mean:</strong> {stats['mean']:.2f} Mg/ha</div>
                    <div class="stat"><strong>Max:</strong> {stats['max']:.2f} Mg/ha</div>
                    <div class="stat"><strong>Min:</strong> {stats['min']:.2f} Mg/ha</div>
                    <div class="stat"><strong>Total:</strong> {stats['total']:.0f} Mg</div>
                """
                
                # Add heatmap if available
                if result['heatmap_path'] and os.path.exists(result['heatmap_path']):
                    rel_path = os.path.relpath(result['heatmap_path'], './outputs')
                    html_content += f"""
                        <div style="margin-top: 15px;">
                            <img src="{rel_path}" alt="{city_name} Heatmap" class="heatmap">
                        </div>
                    """
                
                # Add links to outputs
                html_content += """
                    <div style="margin-top: 15px;">
                        <strong>Outputs:</strong>
                        <ul>
                """
                
                if result['heatmap_path']:
                    rel_path = os.path.relpath(result['heatmap_path'], './outputs')
                    html_content += f'<li><a href="{rel_path}" target="_blank">Heatmap</a></li>'
                
                if result['interactive_map_path']:
                    rel_path = os.path.relpath(result['interactive_map_path'], './outputs')
                    html_content += f'<li><a href="{rel_path}" target="_blank">Interactive Map</a></li>'
                
                if result['pdf_report_path']:
                    rel_path = os.path.relpath(result['pdf_report_path'], './outputs')
                    html_content += f'<li><a href="{rel_path}" target="_blank">PDF Report</a></li>'
                
                html_content += """
                        </ul>
                    </div>
                """
            else:
                html_content += f"<p><strong>Error:</strong> {result['error']}</p>"
            
            html_content += """
                    </div>
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Summary report saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")

if __name__ == "__main__":
    main()