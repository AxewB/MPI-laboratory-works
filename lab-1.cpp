#include <stdio.h> 
#include "mpi.h"
#include <stdlib.h>
int main(int argc, char* argv[]) {
	int ProcRank, ProcSize, RecvRank;
	MPI_Status Status;

	// Инициализация MPI
	MPI_Init(&argc, &argv);

	// Определение количества процессов и ранга текущего процесса
	MPI_Comm_size(MPI_COMM_WORLD, &ProcSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	if (ProcRank == 0) {
		// Процесс с рангом 0 выводит свое сообщение
		printf("\n Hello from process %3d", ProcRank);

		// Начинаем счетчик с 1 из-за вычета исходного процесса 0
	  int i = 1;
		while (i < ProcSize) {
			MPI_Recv(&RecvRank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);

			// Выводим сообщение от полученного процесса
			printf("\n Hello from process %3d", RecvRank);
			i++;
		}
	} else {
		// Процесс с рангом не 0 отправляет сообщение процессу с рангом 0
		MPI_Send(&ProcRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	// Завершение работы MPI
	MPI_Finalize();

	return 0;
}