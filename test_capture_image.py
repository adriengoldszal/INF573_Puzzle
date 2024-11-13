import cv2

# Remplacez par l'URL de votre caméra IP
url = "http://10.222.14.104:8080/video"  # Assurez-vous que c'est bien l'URL du flux vidéo

# Ouvrir le flux vidéo
cap = cv2.VideoCapture(url)
print(type(cap))

if not cap.isOpened():
    print("Erreur : Impossible d'accéder au flux vidéo.")
else:
    while True:
        # Lire l'image du flux vidéo
        ret, frame = cap.read()

        if not ret:
            print("Erreur lors de la lecture du flux vidéo.")
            break

        # Afficher l'image
        cv2.imshow("Flux en direct", frame)

        # Quitter la fenêtre en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
