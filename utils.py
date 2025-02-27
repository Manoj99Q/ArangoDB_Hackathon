import json

def create_data_preview(data, max_items=2, current_depth=0, max_depth=7):
    """
    Create a compact preview of data structures for prompts
    
    This function creates a human-readable preview of complex data structures,
    limiting the depth and number of items shown to keep the output manageable.
    
    Args:
        data: The data structure to preview (list, dict, or other objects)
        max_items: Maximum number of items to show in lists
        current_depth: Current recursion depth (used internally)
        max_depth: Maximum recursion depth before truncating
    
    Returns:
        A simplified version of the data structure suitable for display
    """
    if current_depth > max_depth:
        return "..."
    
    # Handle NetworkX node data (tuple format)
    if isinstance(data, tuple) and len(data) == 2:
        node_id, attributes = data
        return {
            "node_id": node_id,
            "attributes": create_data_preview(attributes, max_items, current_depth+1, max_depth)
        }
        
    # Handle NetworkX NodeAttrDict
    if hasattr(data, 'items'):
        return {str(k): create_data_preview(v, max_items, current_depth+1, max_depth) 
                for k, v in data.items()}

    if isinstance(data, list):
        if not data:
            return []
        sample = data[:min(max_items, len(data))]
        preview = [create_data_preview(item, max_items, current_depth+1, max_depth) for item in sample]
        if len(data) > max_items:
            preview.append(f"... ({len(data) - max_items} more items)")
        return preview
        
    elif isinstance(data, dict):
        return {str(k): create_data_preview(v, max_items, current_depth+1, max_depth) 
                for k, v in data.items()}
                
    # Handle other non-serializable types
    try:
        json.dumps(data)
        return data
    except TypeError:
        return str(data) 