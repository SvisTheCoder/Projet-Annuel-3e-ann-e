#include "ml_api.h"
#include "net_compat.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int receive_all(socket_t socket, void* buffer, int byte_count) {
    char* cursor = buffer;
    int received_total = 0;

    while (received_total < byte_count) {
        int received = recv(
            socket,
            cursor + received_total,
            byte_count - received_total,
            0
        );

        if (received <= 0) {
            return 0;
        }

        received_total += received;
    }

    return 1;
}

static int send_all(
    socket_t socket,
    const void* buffer,
    int byte_count
) {
    const char* cursor = buffer;
    int sent_total = 0;

    while (sent_total < byte_count) {
        int sent = send(
            socket,
            cursor + sent_total,
            byte_count - sent_total,
            0
        );

        if (sent <= 0) {
            return 0;
        }

        sent_total += sent;
    }

    return 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage : %s modele.txt port\n", argv[0]);
        return 1;
    }

    MLModel* model = ml_load(argv[1]);

    if (model == NULL) {
        printf("Erreur modele : %s\n", ml_last_error());
        return 1;
    }

    if (!net_init()) {
        printf("Initialisation reseau impossible.\n");
        ml_free(model);
        return 1;
    }

    int port = atoi(argv[2]);
    socket_t server_socket = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {0};

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons((unsigned short)port);

    if (bind(
            server_socket,
            (struct sockaddr*)&address,
            sizeof(address)) == SOCKET_ERROR
        || listen(server_socket, 5) == SOCKET_ERROR) {
        printf("Impossible de demarrer le serveur.\n");
        CLOSESOCK(server_socket);
        net_cleanup();
        ml_free(model);
        return 1;
    }

    printf("Serveur lance sur le port %d.\n", port);
    printf("Le modele attend %d caracteristiques.\n",
           ml_feature_count(model));

    while (1) {
        socket_t client_socket = accept(server_socket, NULL, NULL);

        if (client_socket == INVALID_SOCKET) {
            continue;
        }

        uint32_t network_feature_count;

        if (!receive_all(
                client_socket,
                &network_feature_count,
                sizeof(network_feature_count))) {
            CLOSESOCK(client_socket);
            continue;
        }

        int feature_count = (int)ntohl(network_feature_count);

        if (feature_count != ml_feature_count(model)) {
            int32_t error_response = htonl(-1);
            send_all(
                client_socket,
                &error_response,
                sizeof(error_response)
            );
            CLOSESOCK(client_socket);
            continue;
        }

        double* values = malloc(sizeof(double) * feature_count);

        if (values == NULL
            || !receive_all(
                client_socket,
                values,
                (int)(sizeof(double) * feature_count))) {
            free(values);
            CLOSESOCK(client_socket);
            continue;
        }

        int prediction = ml_predict(model, values);
        int32_t network_prediction = htonl(prediction);

        send_all(
            client_socket,
            &network_prediction,
            sizeof(network_prediction)
        );

        printf("Prediction envoyee : %d\n", prediction);
        free(values);
        CLOSESOCK(client_socket);
    }

    /* Cette partie n'est atteinte qu'en cas d'arret de la boucle. */
    CLOSESOCK(server_socket);
    net_cleanup();
    ml_free(model);
    return 0;
}
