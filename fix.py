import glob, re

for f in glob.glob('dashboard/views/*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. Fix st.image
    content = content.replace("st.image(wc_img, use_container_width=True, config={'displayModeBar': False})", "st.image(wc_img, use_container_width=True)")
    
    # 2. Fix st.button in ai_lab.py
    content = content.replace("st.button(\"Simpan Jurnal\", use_container_width=True, config={'displayModeBar': False})", "st.button(\"Simpan Jurnal\", use_container_width=True)")
    content = content.replace("col_reset.button(\"Reset Form\", use_container_width=True, config={'displayModeBar': False})", "col_reset.button(\"Reset Form\", use_container_width=True)")
    
    # 3. Add theme=None to all st.plotly_chart that already have config={'displayModeBar': False}
    content = re.sub(r'st\.plotly_chart\(([^,]+),\s*use_container_width=True,\s*config=\{.*?\}\)', r"st.plotly_chart(\1, use_container_width=True, theme=None, config={'displayModeBar': False})", content)
    
    # 4. Add theme=None to all st.plotly_chart that DO NOT have config
    content = re.sub(r'st\.plotly_chart\(([^,]+),\s*use_container_width=True\)', r"st.plotly_chart(\1, use_container_width=True, theme=None)", content)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print("done")
