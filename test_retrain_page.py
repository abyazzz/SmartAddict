#!/usr/bin/env python
"""Test retrain page rendering"""

from app import app

with app.test_client() as client:
    # Login as admin first (need to create test admin session)
    with client.session_transaction() as sess:
        sess['user_id'] = 1  # Assuming admin user_id = 1
        sess['_fresh'] = True
    
    # Test GET /admin/retrain
    print("\n=== Testing GET /admin/retrain ===")
    response = client.get('/admin/retrain')
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.content_type}")
    
    if response.status_code == 200:
        html = response.data.decode('utf-8')
        
        # Check if key elements are present
        checks = {
            'Title': 'Retrain' in html and 'Management' in html,
            'Stats Cards': 'Total Data Session' in html,
            'START RETRAIN Button': 'Start Retrain' in html or 'START RETRAIN' in html,
            'Filter Bar': 'filterGender' in html,
            'Empty State': 'Tidak ada data' in html or 'empty-state' in html,
            'Active Model': 'Active Model' in html or 'active_model_version' in html,
        }
        
        print("\n=== Content Checks ===")
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}: {'FOUND' if result else 'NOT FOUND'}")
        
        # Check for specific button
        if 'btn-start-retrain' in html:
            print("\n✓ Button class 'btn-start-retrain' found in HTML")
        if 'onclick="startRetrain()"' in html:
            print("✓ Button onclick handler found")
            
    elif response.status_code == 302:
        print(f"Redirect to: {response.location}")
        print("(Login required - expected for non-authenticated test)")
    else:
        print(f"Unexpected status code: {response.status_code}")
        print(response.data.decode('utf-8')[:500])

print("\n=== Test Complete ===\n")
