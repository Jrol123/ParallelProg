#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <set>
#include <vector>
#include <mpi.h>
#include <cstring>

using namespace std;

// Функция для чтения содержимого файла и возвращения его в виде строки
string readFileIntoString(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        throw runtime_error("Cannot open the file: " + filename);
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Функция для подсчета уникальных слов из вектора слов
set<string> countUniqueWords(const vector<string> &words)
{
    return set<string>(words.begin(), words.end());
}

// Функция для извлечения слов из содержимого файла
void extractWords(const string &fileContent, vector<string> &words)
{
    string word;
    // Проходим по каждому символу содержимого файла
    for (char ch : fileContent)
    {
        ch = tolower(ch);
        // Если символ - пробел или знак препинания
        if (isspace(ch) || ispunct(ch))
        {
            if (!word.empty())
            {
                words.push_back(word);
                word.clear();
            }
        }
        else
        {
            word += ch;
        }
    }
    // Проверка последнего слова
    if (!word.empty())
    {
        words.push_back(word);
    }
}

// Основная функция программы
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string fileContent;
    vector<string> words;
    int N;

    // Только процесс с рангом 0 будет загружать файл и извлекать слова
    if (rank == 0)
    {
        double start_time = MPI_Wtime();

        fileContent = readFileIntoString("text.txt");
        extractWords(fileContent, words);
        N = words.size();

        double end_time = MPI_Wtime();

        cout << "Sequential count: " << countUniqueWords(words).size() << "\n";
        cout << "Sequential execution time: " << end_time - start_time << endl;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int base_size = N / size;
    int remainder = N % size;
    // Определение локального размера на основе остатка
    int local_size = base_size + (rank < remainder ? 1 : 0);

    vector<string> local_words(local_size);
    if (rank == 0)
    {
        // Распределение слов между процессами с помощью send.
        for (int i = 0; i < N; ++i)
        {
            if (i % size == 0)
            {
                local_words[i / size] = words[i];
            }
            else
            {
                int dest = i % size;

                MPI_Send(words[i].c_str(), words[i].length() + 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (int i = 0; i < local_size; ++i)
        {
            char buffer[100];
            MPI_Recv(buffer, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_words[i] = string(buffer);
        }
    }

    // Собираем уникальные слова локально
    set<string> localUniqueWords = countUniqueWords(local_words);
    vector<string> allUniqueWords;

    double start_time = MPI_Wtime();

    if (rank != 0)
    {
        // Отправка локальных уникальных слов на нулевой процесс
        for (const auto &word : localUniqueWords)
        {
            int length = word.length() + 1;
            MPI_Send(word.c_str(), length, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        int count = localUniqueWords.size();
        MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    else
    {
        int totalWords = localUniqueWords.size();
        for (int i = 1; i < size; ++i)
        {
            int count;
            MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            totalWords += count;
            for (int j = 0; j < count; ++j)
            {
                char buffer[100];
                MPI_Recv(buffer, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                allUniqueWords.emplace_back(buffer);
            }
        }

        for (const auto &word : localUniqueWords)
        {
            allUniqueWords.emplace_back(word);
        }

        double end_time = MPI_Wtime();

        // Подсчет общего числа уникальных слов
        set<string> globalUniqueWords(allUniqueWords.begin(), allUniqueWords.end());
        cout << "Global unique count: " << globalUniqueWords.size() << endl;
        cout << "Parallel execution time: " << end_time - start_time << endl;
    }

    MPI_Finalize();
    return 0;
}
