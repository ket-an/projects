import streamlit as st
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import io

# set page config
st.set_page_config(page_title="JMeter Report Generator", layout="wide")

st.title("📊 JMeter Report Generator")
st.markdown("Upload your JMeter Summary Excel (.xlsx) to generate the PDF report.")

# sidebar for metadata
st.sidebar.header("Test Metadata")
project_name = st.sidebar.text_input("Module / Project Name", value="Selfcare FAQ module")
run_id = st.sidebar.text_input("Run ID", value="2241")

# Use text_input for times to allow flexibility
start_time_input = st.sidebar.text_input("Start Time", value="")
end_time_input = st.sidebar.text_input("End Time", value="")
users_input = st.sidebar.text_input("Active Users (Threads)", value="10")
duration_input = st.sidebar.text_input("Duration", value="5 Mins")

st.sidebar.subheader("Resource Utilization (Results)")
cpu_res = st.sidebar.text_input("CPU Usage (Result)", value="~27%")
mem_res = st.sidebar.text_input("Memory Usage (Result)", value="~60%")

st.sidebar.subheader("PT Configuration")
num_pods = st.sidebar.text_input("Number of pods", value="2")
cpu_req = st.sidebar.text_input("CPU request", value="500m")
cpu_lim = st.sidebar.text_input("CPU limit", value="1000m")
mem_req = st.sidebar.text_input("Memory request", value="512Mi")
mem_lim = st.sidebar.text_input("Memory limit", value="1Gi")

# file uploader
uploaded_file = st.file_uploader("Upload JMeter Summary Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load header=1 to capture the second row as headers
        df = pd.read_excel(uploaded_file, header=1)
        
        # --- Fix Column Names (Merged Headers handling) ---
        new_columns = df.columns.tolist()
        
        # Index 0: API Name
        if "Unnamed" in str(new_columns[0]): new_columns[0] = "API Name"
        
        # Look for Throughput and Error % (which might be Unnamed in row 1 if they were in row 0)
        # Typically they are at the end. Let's try to map by index if column count matches expected
        # Expected: API, Duration, Total, Success, Failed, Avg, Max, Min, StdDev, P50, P75, P90, P95, P99, Throughput, Error %
        
        if len(new_columns) >= 15:
            if "Unnamed" in str(new_columns[14]): new_columns[14] = "Throughput"
            if len(new_columns) > 15 and "Unnamed" in str(new_columns[15]): new_columns[15] = "Error %"
            
        df.columns = new_columns
        
        # Normalize
        if 'API Name' not in df.columns:
            df.rename(columns={df.columns[0]: 'API Name'}, inplace=True)
            
        # We need Failed count to calculate weighted aggregate error %
        # Check for 'Failed' column or map it from index 4
        if 'Failed' not in df.columns and len(df.columns) > 4:
             # Just in case the name is different
             # But let's assume if it's not found we try index 4
             pass 

        required_cols = ['API Name', 'Total', 'Failed', 'Avg', 'P90', 'P95', 'P99', 'Error %', 'Throughput']
        
        # Cleanup column names
        df.columns = df.columns.str.strip()
        
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
             # Try stricter mapping if columns are missing (e.g. maybe 'Failed' is named differently)
             st.error(f"Missing columns: {missing}. Please check format.")
             st.write("Current Columns:", df.columns.tolist())
        else:
            st.success("File Processed.")
            
            # --- Filter out existing TOTAL row if present to recalculate ---
            # Some JMetere reports have a Total row, some don't. 
            # We filter it out to avoid double counting, then add our own calculated Total.
            data_df = df[df['API Name'].astype(str).str.upper() != 'TOTAL'].copy()
            
            # Convert numeric columns
            cols_to_numeric = ['Total', 'Failed', 'Avg', 'P90', 'P95', 'P99', 'Throughput']
            for col in cols_to_numeric:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce').fillna(0)
            
            # Clean Error % (remove % sign if string)
            # Instead of relying on the 'Error %' column from Excel which has formatting issues (fractions vs percent),
            # matches user request to "apply logic... to make value upto two decimal places"
            # We explicitly RECALCULATE Error % from Failed and Total counts.
            
            # Ensure no division by zero
            data_df['Error %'] = data_df.apply(lambda row: (row['Failed'] / row['Total'] * 100) if row['Total'] > 0 else 0, axis=1)
            
            # Round to 2 decimals immediately as requested
            data_df['Error %'] = data_df['Error %'].round(2)
                 
            # --- Calculate Totals ---
            total_samples = data_df['Total'].sum()
            total_failed = data_df['Failed'].sum()
            total_tps = data_df['Throughput'].sum()
            
            # User request: "after changing the values to 2 decimal places then calculate total Error %"
            # This implies a simple SUM of the rounded values, not a weighted average.
            # Example provided: 0.22 + 0.22 = 0.44% (Sum).
            total_error_pct = data_df['Error %'].sum()
            
            # Create Total Row
            total_row = {
                'API Name': 'TOTAL',
                'Total': int(total_samples),
                'Failed': int(total_failed),
                'Avg': '',  # User asked for Total Samples, TPS and Error % ONLY
                'P90': '',
                'P95': '',
                'P99': '',
                'Error %': total_error_pct,
                'Throughput': total_tps
            }
            
            # Convert to DataFrame
            # data_df has all columns, we append total
            df_display = pd.concat([data_df, pd.DataFrame([total_row])], ignore_index=True)
            
            # --- Formatting ---
            # Rename columns for final output
            df_display.rename(columns={'Total': 'Samples', 'Throughput': 'TPS'}, inplace=True)
            
            # Apply display formatting (Decimal places)
            # We do this on a copy to keep numeric consistency if needed, but for display/PDF strings are fine
            
            formatted_df = df_display.copy()
            
            # Format TPS and Error % to 2 decimals
            formatted_df['TPS'] = formatted_df['TPS'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            formatted_df['Error %'] = formatted_df['Error %'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
            
            # Format Ints
            for col in ['Samples', 'Avg', 'P90', 'P95', 'P99']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(x)}" if x != '' and isinstance(x, (int, float)) else x)

            # Select final columns
            final_cols = ['API Name', 'Samples', 'Avg', 'P90', 'P95', 'P99', 'Error %', 'TPS']
            final_table = formatted_df[final_cols]
            
            st.subheader("Preview Report Data")
            st.dataframe(final_table, use_container_width=True)
            
            # --- PDF Generation ---
            
            class PDF(FPDF):
                def header(self):
                    pass
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

            def generate_custom_pdf(df_data, meta):
                pdf = PDF()
                pdf.add_page()
                
                # --- Title ---
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 8, meta['project'], ln=1, align='L')
                pdf.ln(2)
                
                # --- Metadata Rows ---
                pdf.set_font('Arial', 'B', 10)
                
                # Row 1: Start/End Time
                pdf.cell(0, 6, f"Start Time: {meta['start']}  End Time: {meta['end']}", ln=1)
                
                # Row 2: Stats (Yellow)
                pdf.set_fill_color(255, 255, 204) # Light Yellow
                info_line = f"Users: {meta['users']}   Duration: {meta['duration']}   RunId: {meta['run_id']}   CPU usage: {meta['cpu_res']}   Memory usage: {meta['mem_res']}"
                pdf.cell(0, 6, info_line, ln=1, fill=True)
                
                # Row 3: PT Config description (New)
                # "Configuration used for PT"
                pdf.ln(2)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, "Configuration used for PT:", ln=1)
                
                # Row 4: Config Details
                # Pods, CPU Req/Lim, Mem Req/Lim
                pdf.set_font('Arial', '', 10)
                config_line = (f"Number of pods: {meta['pods']}   "
                               f"CPU req/lim: {meta['cpu_req']}/{meta['cpu_lim']}   "
                               f"Mem req/lim: {meta['mem_req']}/{meta['mem_lim']}")
                pdf.cell(0, 6, config_line, ln=1)
                
                pdf.ln(4)

                # --- Table ---
                col_widths = [60, 20, 15, 15, 15, 15, 20, 20]
                headers = ['API Name', 'Samples', 'Avg', 'P90', 'P95', 'P99', 'Error %', 'TPS']
                
                # Header
                pdf.set_font('Arial', 'B', 9)
                pdf.set_fill_color(220, 220, 220) # Gray
                for i, h in enumerate(headers):
                    pdf.cell(col_widths[i], 8, h, 1, 0, 'C', True)
                pdf.ln()
                
                # Rows
                pdf.set_font('Arial', '', 9)
                for _, row in df_data.iterrows():
                    is_total = str(row['API Name']) == 'TOTAL'
                    if is_total:
                        pdf.set_font('Arial', 'B', 9)
                    else:
                        pdf.set_font('Arial', '', 9)
                    
                    # Truncate Name
                    name = str(row['API Name'])
                    if len(name) > 35: name = name[:32] + "..."
                    
                    pdf.cell(col_widths[0], 7, name, 1)
                    pdf.cell(col_widths[1], 7, str(row['Samples']), 1, 0, 'C')
                    pdf.cell(col_widths[2], 7, str(row['Avg']), 1, 0, 'C')
                    pdf.cell(col_widths[3], 7, str(row['P90']), 1, 0, 'C')
                    pdf.cell(col_widths[4], 7, str(row['P95']), 1, 0, 'C')
                    pdf.cell(col_widths[5], 7, str(row['P99']), 1, 0, 'C')
                    pdf.cell(col_widths[6], 7, str(row['Error %']), 1, 0, 'C')
                    pdf.cell(col_widths[7], 7, str(row['TPS']), 1, 0, 'C')
                    pdf.ln()
                
                return bytes(pdf.output(dest='S'))

            # Meta Data
            meta_info = {
                'project': project_name,
                'run_id': run_id,
                'start': start_time_input,
                'end': end_time_input,
                'users': users_input,
                'duration': duration_input,
                'cpu_res': cpu_res,
                'mem_res': mem_res,
                'pods': num_pods,
                'cpu_req': cpu_req,
                'cpu_lim': cpu_lim,
                'mem_req': mem_req,
                'mem_lim': mem_lim
            }

            if st.button("Generate PDF Report"):
                pdf_bytes = generate_custom_pdf(final_table, meta_info)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="jmeter_pt_report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
