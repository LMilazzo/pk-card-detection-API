Launch EC2 instance

SSH into instance 

Using pem file and public DNS

sudo apt-get update

sudo apt install -y python3 pip nginx

create nginx config file in /etc/nginx/sites-enabled/fastapi_nginx

copy paste contents here adding your public ipv4

esc :wq enter

sudo service nginx restart

git clone repo

cd to repo 

sudo apt install python3.12-venv

python3 -m venv env_name

source env/bin/activate

pip install -r dependencies.txt

sudo apt update
sudo apt install -y libgl1

pip intall python-multipart

leave ssh with exit

copy over model pt with:
   scp -i "pem_path" "pt_path" ubuntu@DNS/home/ubuntu/

ssh

sudo apt install git-lfs

git lfs pull

uvicorn main:app



