from onvif import ONVIFCamera

IP = "172.16.1.13"
PORT = 554
USER = "admin"
PASS = "test12345"

cam = ONVIFCamera(IP, PORT, USER, PASS)
media = cam.create_media_service()
profiles = media.GetProfiles()

for p in profiles:
    uri = media.GetStreamUri({
        'StreamSetup': {
            'Stream': 'RTP-Unicast',
            'Transport': {'Protocol': 'RTSP'}
        },
        'ProfileToken': p.token
    })
    print(uri.Uri)
