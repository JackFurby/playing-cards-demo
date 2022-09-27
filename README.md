# playing cards demo

Please see the Wiki for details about setup and running the application.




CBM demo for playing card recognition


sudo apt-get install redis-server

flask --app playing_cards --debug run


celery -A playing_cards.celery worker


docker build --tag playing-cards-demo .

docker run -p 5000:5000 playing-cards-demo
