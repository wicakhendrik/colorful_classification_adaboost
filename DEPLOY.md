# DEPLOY.md — Deploy ke VPS

Dokumen ini untuk **manusia dan AI agent**. Ikuti urut dari atas. Setiap langkah punya perintah verifikasi — jangan lanjut kalau verifikasinya gagal.

## 0. Ringkasan

| Item | Nilai |
|---|---|
| Jenis app | Django 3.1 (WSGI) + gunicorn, satu container |
| Runtime | **Python 3.8** (wajib, dibungkus Docker) |
| Port container | `8000` (HTTP) |
| Database | Tidak dipakai (app tanpa model). `db.sqlite3` hanya bawaan Django |
| State yang perlu dijaga | Folder `media/` (gambar upload) → Docker volume `media` |
| Static files | Di-serve WhiteNoise dari dalam container, `collectstatic` jalan saat build |
| Health check | `GET /` → HTTP 200 |
| Kebutuhan server | Linux x86_64/arm64, ≥ 1 vCPU, **≥ 2 GB RAM** (1 GB + swap minimal), ≥ 3 GB disk |

Pakai **Docker** — jangan install Python 3.8 manual di VPS kecuali terpaksa (lihat bagian 8).

## 1. Prasyarat di VPS

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"      # logout/login ulang setelah ini
docker --version && docker compose version
sudo apt-get install -y git
```

Buka firewall hanya untuk 22, 80, 443 (port 8000 **tidak** perlu dibuka ke publik):

```bash
sudo ufw allow OpenSSH && sudo ufw allow 80,443/tcp && sudo ufw enable
```

## 2. Ambil kode

```bash
sudo mkdir -p /opt/colorful && sudo chown "$USER" /opt/colorful
git clone <URL_REPO> /opt/colorful
cd /opt/colorful
```

## 3. Konfigurasi `.env`

```bash
cp .env.example .env
SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")
sed -i "s|^DJANGO_SECRET_KEY=.*|DJANGO_SECRET_KEY=${SECRET}|" .env
```

Lalu edit `.env`:

```ini
APP_BIND=127.0.0.1                       # hanya bisa diakses lewat reverse proxy
APP_PORT=8000
DJANGO_DEBUG=False
DJANGO_SECRET_KEY=<hasil generate di atas>
DJANGO_ALLOWED_HOSTS=colorful.example.com,<IP_VPS>
DJANGO_CSRF_TRUSTED_ORIGINS=colorful.example.com
```

Catatan penting:
- `DJANGO_ALLOWED_HOSTS` **harus** berisi domain/IP yang dipakai browser, kalau tidak Django membalas `400 Bad Request`.
- Django 3.1 memakai host tanpa skema untuk `CSRF_TRUSTED_ORIGINS` (`colorful.example.com`, bukan `https://…`).
- Jangan commit `.env` (sudah ada di `.gitignore`).

## 4. Build & jalankan

```bash
docker compose up -d --build
```

Build pertama ±3–6 menit (download numpy/scipy/sklearn/opencv). Image ±800 MB.

**Verifikasi:**

```bash
docker compose ps                                  # status: running
docker compose logs --tail 20 web                  # harus ada "Listening at: http://0.0.0.0:8000"
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/                 # 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/static/style.css  # 200
```

Bila RAM kecil dan build `Killed`, tambah swap:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

## 5. Reverse proxy + HTTPS

Arahkan DNS A-record domain ke IP VPS dulu. Pilih salah satu.

### Opsi A — Caddy (paling simpel, HTTPS otomatis)

```bash
sudo apt-get install -y caddy
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
colorful.example.com {
    request_body {
        max_size 20MB
    }
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            response_header_timeout 120s
        }
    }
}
EOF
sudo systemctl reload caddy
```

### Opsi B — Nginx + Certbot

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo tee /etc/nginx/sites-available/colorful >/dev/null <<'EOF'
server {
    listen 80;
    server_name colorful.example.com;
    client_max_body_size 20m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;   # klasifikasi bisa makan 10-20 detik
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/colorful /etc/nginx/sites-enabled/colorful
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d colorful.example.com
```

> ⚠️ `proxy_read_timeout`/`response_header_timeout` **harus ≥ 120s**. Default nginx 60s sudah cukup untuk kebanyakan kasus, tapi VPS lambat bisa lebih lama; gunicorn sendiri di-set timeout 120s.

**Verifikasi dari luar:**

```bash
curl -I https://colorful.example.com/        # HTTP/2 200
```

Lalu uji klasifikasi end-to-end (ambil CSRF token dulu):

```bash
cd /opt/colorful
curl -s -c /tmp/cj https://colorful.example.com/ -o /tmp/p.html
TOK=$(grep -o 'csrfmiddlewaretoken" value="[^"]*' /tmp/p.html | sed 's/.*value="//')
curl -s -b /tmp/cj -F csrfmiddlewaretoken=$TOK -F input_img=@media/yes.png https://colorful.example.com/ | grep "<p>"
# expect: <p>Colorful</p>   (media/no.png → <p>Nocolorful</p>)
```

## 6. Update / redeploy

```bash
cd /opt/colorful
git pull
docker compose up -d --build        # rebuild hanya layer yang berubah
docker image prune -f               # bersihkan image lama
```

Downtime hanya beberapa detik saat container diganti.

## 7. Operasional

```bash
docker compose logs -f web          # log
docker compose restart web          # restart
docker compose down                 # stop (volume media tetap ada)
docker compose down -v              # ⚠️ stop + HAPUS volume media (upload hilang)
```

**Backup upload:**

```bash
docker run --rm -v colorful_media:/data -v "$PWD":/backup alpine \
  tar czf /backup/media-$(date +%F).tgz -C /data .
```

(Nama volume = `<nama-folder-project>_media`; cek dengan `docker volume ls`. Bila folder `/opt/colorful` → `colorful_media`.)

**Bersihkan upload lama** (folder `media/` tidak pernah dibersihkan otomatis):

```bash
docker compose exec web find media -type f -mtime +30 ! -name 'yes.png' ! -name 'no.png' ! -name '9.png' ! -name '11.png' ! -name 'no_llvam8b.png' -delete
```

> Catatan: volume `media` yang kosong pertama kali di-mount akan diisi dari isi image (gambar contoh) oleh Docker.

## 8. Alternatif tanpa Docker (tidak direkomendasikan)

Hanya jika Docker tidak bisa dipakai. Butuh **Python 3.8** — Ubuntu 22.04/24.04 tidak menyediakannya di repo standar:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv libgl1 libglib2.0-0
cd /opt/colorful
python3.8 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python manage.py collectstatic --noinput
```

Systemd unit `/etc/systemd/system/colorful.service`:

```ini
[Unit]
Description=Colorful classification (gunicorn)
After=network.target

[Service]
WorkingDirectory=/opt/colorful
EnvironmentFile=/opt/colorful/.env
ExecStart=/opt/colorful/.venv/bin/gunicorn colorful_classification_adaboost.wsgi:application --bind 127.0.0.1:8000 --workers 2 --timeout 120
Restart=always
User=www-data

[Install]
WantedBy=multi-user.target
```

```bash
sudo chown -R www-data /opt/colorful/media
sudo systemctl daemon-reload && sudo systemctl enable --now colorful
```

(Aplikasi harus dijalankan dengan working directory = root project, karena `views.py` membaca `static/bbt_pred.csv` lewat path relatif.)

## 9. Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `400 Bad Request` | Host tidak ada di `DJANGO_ALLOWED_HOSTS` | Tambahkan domain/IP, `docker compose up -d` |
| `403 CSRF verification failed` | Proxy tidak meneruskan `Host`, atau domain belum di `DJANGO_CSRF_TRUSTED_ORIGINS` | Pakai config proxy di atas; set variabel lalu restart |
| `502 Bad Gateway` | Container mati / port salah | `docker compose ps`, `docker compose logs web`; pastikan `APP_PORT` sama dengan `proxy_pass` |
| `504` / worker timeout saat upload | Proxy timeout < waktu K-Means | Naikkan timeout proxy ≥ 120s |
| Gunicorn `WORKER TIMEOUT` / `Killed` (OOM) | RAM kurang | Tambah swap / RAM, kurangi `--workers` ke 1 di `Dockerfile` |
| `ImportError: libGL.so.1` | Jalan tanpa Docker, lib sistem kurang | `apt install libgl1 libglib2.0-0` |
| `pip install` gagal build numpy/opencv | Python bukan 3.8/3.9 | Pakai Docker atau Python 3.8 |
| Error 500 saat upload | File bukan gambar valid | Upload PNG/JPG (belum ada validasi di kode) |
| Gambar upload 404 | Volume `media` hilang / nama file tak ada | Cek `docker compose exec web ls media` |
| CSS tidak termuat | `collectstatic` tidak jalan | `docker compose build --no-cache` |

## 10. Checklist untuk AI agent

Jalankan berurutan, hentikan dan laporkan bila ada yang gagal:

1. [ ] `docker --version && docker compose version` sukses.
2. [ ] Repo ada di `/opt/colorful`, `git status` bersih (kecuali `.env`).
3. [ ] `.env` ada, `DJANGO_DEBUG=False`, `DJANGO_SECRET_KEY` bukan `change-me`, `DJANGO_ALLOWED_HOSTS` berisi domain & IP.
4. [ ] `docker compose up -d --build` sukses dan `docker compose ps` menunjukkan `running`.
5. [ ] `curl http://127.0.0.1:8000/` → `200`.
6. [ ] Upload `media/yes.png` → body mengandung `Colorful`; `media/no.png` → `Nocolorful` (tes end-to-end di bagian 5).
7. [ ] Reverse proxy aktif dan `curl -I https://<domain>/` → `200`.
8. [ ] Port 8000 tidak terbuka publik: dari luar VPS, `curl http://<IP_VPS>:8000` harus gagal.

**Jangan:** mengubah `requirements.txt` ke versi baru (model `.pkl` dibuat dengan scikit-learn 0.24.0 dan akan gagal/berperilaku beda), menjalankan `docker compose down -v` tanpa backup, atau mengaktifkan `DJANGO_DEBUG=True` di server publik.
