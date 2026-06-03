import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
from pathlib import Path
import io

with app.app_context():
    from smartaddict.extensions import db
    from smartaddict.models.user import User
    from smartaddict.models.predict_user_session import PredictUserSession

    # create test user if not exists
    user = User.query.filter_by(username='e2e_test_user').first()
    if not user:
        user = User(username='e2e_test_user', role='user')
        user.set_password('password123')
        db.session.add(user)
        db.session.commit()

    # clear existing predict sessions for user
    PredictUserSession.query.filter_by(user_id=user.id).delete()
    db.session.commit()

    # seed 49 rows
    for i in range(49):
        s = PredictUserSession(
            user_id=user.id,
            age=20 + (i % 10),
            gender=1,
            daily_screen_time_hours=2.0,
            social_media_hours=1.0,
            gaming_hours=0.5,
            work_study_hours=6.0,
            sleep_hours=7.0,
            notifications_per_day=5,
            app_opens_per_day=10,
            weekend_screen_time=3.0,
            result='Rendah',
        )
        db.session.add(s)
    db.session.commit()

    print('Seeded sessions:', PredictUserSession.query.count())

# Build CSV with 2 rows, 10 numeric columns
csv_content = '\n'.join([','.join(['1']*10), ','.join(['2']*10)])
print('CSV payload:\n', csv_content)

client = app.test_client()
# register/login via test client
resp = client.post('/login', data={'username':'e2e_test_user','password':'password123'}, follow_redirects=True)
print('/login status', resp.status_code)

data = {
    'csv_submit': '1',
}
file_tuple = (io.BytesIO(csv_content.encode('utf-8')), 'data.csv')

# list status files before and after to detect newly created job
status_dir = Path('instance') / 'retrain_statuses'
before_files = set()
if status_dir.exists():
    before_files = set(p.name for p in status_dir.iterdir() if p.suffix == '.json')

resp = client.post('/predict', data={'csv_file': file_tuple, 'csv_submit': '1'}, content_type='multipart/form-data', follow_redirects=True)
print('/predict post status', resp.status_code)

after_files = set()
if status_dir.exists():
    after_files = set(p.name for p in status_dir.iterdir() if p.suffix == '.json')
new = sorted(after_files - before_files)
print('new retrain status files:', new)

with app.app_context():
    print('PredictUserSession count after post:', PredictUserSession.query.count())

print('Done')
