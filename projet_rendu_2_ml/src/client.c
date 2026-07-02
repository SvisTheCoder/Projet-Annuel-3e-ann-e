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
    if (argc < 4) {
        printf("Usage : %s ip port nombre_caracteristiques\n", argv[0]);
        return 1;
    }

    if (!net_init()) {
        printf("Initialisation reseau impossible.\n");
        return 1;
    }

    int port = atoi(argv[2]);
    int feature_count = atoi(argv[3]);
    double* values = malloc(sizeof(double) * feature_count);

    if (values == NULL) {
        printf("Allocation impossible.\n");
        net_cleanup();
        return 1;
    }

    for (int feature = 0; feature < feature_count; feature++) {
        printf("x[%d] : ", feature);
        scanf("%lf", &values[feature]);
    }

    socket_t socket_client = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address = {0};

    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short)port);

    if (inet_pton(AF_INET, argv[1], &address.sin_addr) != 1
        || connect(
            socket_client,
            (struct sockaddr*)&address,
            sizeof(address)) == SOCKET_ERROR) {
        printf("Connexion impossible.\n");
        free(values);
        CLOSESOCK(socket_client);
        net_cleanup();
        return 1;
    }

    uint32_t network_feature_count = htonl((uint32_t)feature_count);

    if (!send_all(
            socket_client,
            &network_feature_count,
            sizeof(network_feature_count))
        || !send_all(
            socket_client,
            values,
            (int)(sizeof(double) * feature_count))) {
        printf("Envoi impossible.\n");
        free(values);
        CLOSESOCK(socket_client);
        net_cleanup();
        return 1;
    }

    int32_t network_prediction;

    if (receive_all(
            socket_client,
            &network_prediction,
            sizeof(network_prediction))) {
        int prediction = ntohl(network_prediction);

        if (prediction < 0) {
            printf("Le serveur a refuse la donnee.\n");
        } else {
            printf("Classe predite : %d\n", prediction);
        }
    } else {
        printf("Aucune reponse du serveur.\n");
    }

    free(values);
    CLOSESOCK(socket_client);
    net_cleanup();
    return 0;
}
