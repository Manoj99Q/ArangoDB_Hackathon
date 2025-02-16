import gradio as gr

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>D3.js Bar Chart</title>
  <!-- Load D3.js from CDN -->
  <script src="https://d3js.org/d3.v6.min.js"></script>
  <style>
    body { font-family: sans-serif; }
    .bar { fill: steelblue; }
  </style>
</head>
<body>
  <h1>D3.js Bar Chart</h1>
  <div id="chart"></div>
  <script>
    // Create the bar chart
    var data = [30, 86, 168, 281, 303, 365];
    var width = 500;
    var height = 300;
    
    var svg = d3.select("#chart")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
    
    svg.selectAll("rect")
       .data(data)
       .enter().append("rect")
       .attr("width", 40)
       .attr("height", function(d) { return d; })
       .attr("x", function(d, i) { return i * 45; })
       .attr("y", function(d) { return height - d; })
       .attr("fill", "steelblue");
  </script>
</body>
</html>
"""
import base64

encoded_html = base64.b64encode(html_code.encode("utf-8")).decode("utf-8")
iframe_html = f'<iframe src="data:text/html;base64,{encoded_html}" width="600" height="400" frameborder="0"></iframe>'


iface = gr.Interface(fn=lambda: iframe_html, inputs=[], outputs=gr.HTML())
iface.launch()
