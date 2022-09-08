# playing_cards_demo
CBM demo for playing card recognition


sudo apt-get install redis-server

flask --app playing_cards --debug run


celery -A playing_cards.celery worker
