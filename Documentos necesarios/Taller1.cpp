#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

using namespace std;

class Libro
{
private:
    string isbn;
    string titulo;
    string autor;
    int anioPublicacion;
    string categoria;
    float precio;

public:
    Libro(string _isbn = "", string _titulo = "", string _autor = "",
          int _anioPublicacion = 0, string _categoria = "", float _precio = 0.0)
    {
        isbn = _isbn;
        titulo = _titulo;
        autor = _autor;
        anioPublicacion = _anioPublicacion;
        categoria = _categoria;
        precio = _precio;
    }

    // Getters
    string getIsbn() const { return isbn; }
    string getTitulo() const { return titulo; }
    string getAutor() const { return autor; }
    int getAnioPublicacion() const { return anioPublicacion; }
    string getCategoria() const { return categoria; }
    float getPrecio() const { return precio; }

    // Setters
    void setIsbn(string _isbn) { isbn = _isbn; }
    void setTitulo(string _titulo) { titulo = _titulo; }
    void setAutor(string _autor) { autor = _autor; }
    void setAnioPublicacion(int _anioPublicacion) { anioPublicacion = _anioPublicacion; }
    void setCategoria(string _categoria) { categoria = _categoria; }
    void setPrecio(float _precio) { precio = _precio; }
};

class SistemaBiblioteca
{
private:
    vector<Libro> libros;

public:
    void limpiarBuffer()
    {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
    void adicionarLibro()
    {
        string isbn, titulo, autor, categoria;
        int anio;
        float precio;

        cout << "\nIngrese los datos del libro:\n";
        cout << "ISBN: ";
        getline(cin, isbn);
        cout << "Título: ";
        getline(cin, titulo);
        cout << "Autor: ";
        getline(cin, autor);
        cout << "Año de publicación: ";
        cin >> anio;
        limpiarBuffer();
        cout << "Categoría: ";
        getline(cin, categoria);
        cout << "Precio: ";
        cin >> precio;
        limpiarBuffer();

        Libro nuevoLibro(isbn, titulo, autor, anio, categoria, precio);
        libros.push_back(nuevoLibro);
        cout << "\nLibro agregado exitosamente!\n";
    }

    void modificarLibro()
    {
        if (libros.empty())
        {
            cout << "\nNo hay libros registrados.\n";
            return;
        }

        string isbnBuscado;
        cout << "\nIngrese el ISBN del libro a modificar: ";
        getline(cin, isbnBuscado);

        for (auto &libro : libros)
        {
            if (libro.getIsbn() == isbnBuscado)
            {
                cout << "\nIngrese los nuevos datos del libro:\n";
                string titulo, autor, categoria;
                int anio;
                float precio;

                cout << "Título: ";
                getline(cin, titulo);
                libro.setTitulo(titulo);

                cout << "Autor: ";
                getline(cin, autor);
                libro.setAutor(autor);

                cout << "Año de publicación: ";
                cin >> anio;
                limpiarBuffer();
                libro.setAnioPublicacion(anio);

                cout << "Categoría: ";
                getline(cin, categoria);
                libro.setCategoria(categoria);

                cout << "Precio: ";
                cin >> precio;
                limpiarBuffer();
                libro.setPrecio(precio);

                cout << "\nLibro modificado exitosamente!\n";
                return;
            }
        }
        cout << "\nLibro no encontrado.\n";
    }

    void eliminarLibro()
    {
        if (libros.empty())
        {
            cout << "\nNo hay libros registrados.\n";
            return;
        }

        string isbnBuscado;
        cout << "\nIngrese el ISBN del libro a eliminar: ";
        getline(cin, isbnBuscado);

        auto it = remove_if(libros.begin(), libros.end(),
                            [isbnBuscado](const Libro &libro)
                            { return libro.getIsbn() == isbnBuscado; });

        if (it != libros.end())
        {
            libros.erase(it, libros.end());
            cout << "\nLibro eliminado exitosamente!\n";
        }
        else
        {
            cout << "\nLibro no encontrado.\n";
        }
    }

    void submenuListados()
    {
        int opcion;
        do
        {
            cout << "\n--- Submenú Listados ---\n";
            cout << "1. Listar todos los libros\n";
            cout << "2. Listar libros por categoría\n";
            cout << "3. Listar libros por año\n";
            cout << "4. Volver al menú principal\n";
            cout << "Seleccione una opción: ";
            cin >> opcion;
            limpiarBuffer();

            switch (opcion)
            {
            case 1:
                listarTodos();
                break;
            case 2:
                listarPorCategoria();
                break;
            case 3:
                listarPorAnio();
                break;
            case 4:
                cout << "Volviendo al menú principal...\n";
                break;
            default:
                cout << "Opción inválida.\n";
            }
        } while (opcion != 4);
    }

    void listarTodos()
    {
        if (libros.empty())
        {
            cout << "\nNo hay libros registrados.\n";
            return;
        }

        cout << "\n--- Lista de todos los libros ---\n";
        for (const auto &libro : libros)
        {
            mostrarLibro(libro);
        }
    }

    void listarPorCategoria()
    {
        if (libros.empty())
        {
            cout << "\nNo hay libros registrados.\n";
            return;
        }

        string categoriaBuscada;
        cout << "\nIngrese la categoría a buscar: ";
        getline(cin, categoriaBuscada);

        cout << "\n--- Libros de la categoría " << categoriaBuscada << " ---\n";
        bool encontrados = false;
        for (const auto &libro : libros)
        {
            if (libro.getCategoria() == categoriaBuscada)
            {
                mostrarLibro(libro);
                encontrados = true;
            }
        }
        if (!encontrados)
        {
            cout << "No se encontraron libros en esta categoría.\n";
        }
    }

    void listarPorAnio()
    {
        if (libros.empty())
        {
            cout << "\nNo hay libros registrados.\n";
            return;
        }

        int anioBuscado;
        cout << "\nIngrese el año a buscar: ";
        cin >> anioBuscado;
        limpiarBuffer();

        cout << "\n--- Libros del año " << anioBuscado << " ---\n";
        bool encontrados = false;
        for (const auto &libro : libros)
        {
            if (libro.getAnioPublicacion() == anioBuscado)
            {
                mostrarLibro(libro);
                encontrados = true;
            }
        }
        if (!encontrados)
        {
            cout << "No se encontraron libros de este año.\n";
        }
    }

    void mostrarEstadisticas()
    {
        if (libros.empty())
        {
            cout << "\nNo hay libros registrados para mostrar estadísticas.\n";
            return;
        }

        // 1. Precio promedio de los libros
        float precioTotal = 0;
        for (const auto &libro : libros)
        {
            precioTotal += libro.getPrecio();
        }
        float precioPromedio = precioTotal / libros.size();

        // 2. Cantidad de libros por categoría
        vector<pair<string, int>> librosPorCategoria;
        for (const auto &libro : libros)
        {
            string categoria = libro.getCategoria();
            auto it = find_if(librosPorCategoria.begin(), librosPorCategoria.end(),
                              [categoria](const pair<string, int> &p)
                              { return p.first == categoria; });

            if (it != librosPorCategoria.end())
            {
                it->second++;
            }
            else
            {
                librosPorCategoria.push_back({categoria, 1});
            }
        }

        // 3. Libro más antiguo
        int anioMasAntiguo = libros[0].getAnioPublicacion();
        for (const auto &libro : libros)
        {
            if (libro.getAnioPublicacion() < anioMasAntiguo)
            {
                anioMasAntiguo = libro.getAnioPublicacion();
            }
        }

        // Mostrar estadísticas
        cout << "\n--- Estadísticas ---\n";
        cout << "1. Precio promedio de los libros: $" << precioPromedio << endl;
        cout << "2. Cantidad de libros por categoría:\n";
        for (const auto &par : librosPorCategoria)
        {
            cout << "   - " << par.first << ": " << par.second << " libros\n";
        }
        cout << "3. Año del libro más antiguo: " << anioMasAntiguo << endl;
    }

    void mostrarAcercaDe()
    {
        cout << "\n--- Acerca de ---\n";
        cout << "Desarrollado por:\n";
        cout << "12345 - Juan Pérez\n";
        cout << "67890 - María García\n";
    }

private:
    void mostrarLibro(const Libro &libro)
    {
        cout << "\nISBN: " << libro.getIsbn() << endl;
        cout << "Título: " << libro.getTitulo() << endl;
        cout << "Autor: " << libro.getAutor() << endl;
        cout << "Año: " << libro.getAnioPublicacion() << endl;
        cout << "Categoría: " << libro.getCategoria() << endl;
        cout << "Precio: $" << libro.getPrecio() << endl;
        cout << "------------------------\n";
    }
};

int main()
{
    SistemaBiblioteca sistema;
    int opcion;

    do
    {
        cout << "\n=== SISTEMA DE GESTIÓN DE BIBLIOTECA ===\n";
        cout << "1. Adicionar libro\n";
        cout << "2. Modificar libro\n";
        cout << "3. Eliminar libro\n";
        cout << "4. Listados\n";
        cout << "5. Estadísticas\n";
        cout << "6. Acerca de\n";
        cout << "7. Salir\n";
        cout << "Seleccione una opción: ";
        cin >> opcion;
        sistema.limpiarBuffer();

        switch (opcion)
        {
        case 1:
            sistema.adicionarLibro();
            break;
        case 2:
            sistema.modificarLibro();
            break;
        case 3:
            sistema.eliminarLibro();
            break;
        case 4:
            sistema.submenuListados();
            break;
        case 5:
            sistema.mostrarEstadisticas();
            break;
        case 6:
            sistema.mostrarAcercaDe();
            break;
        case 7:
            cout << "\n¡Gracias por usar el sistema!\n";
            break;
        default:
            cout << "\nOpción inválida. Por favor intente de nuevo.\n";
        }
    } while (opcion != 7);

    return 0;
}