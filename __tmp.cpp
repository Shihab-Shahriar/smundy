template <typename T, typename U>
void random_init(T& t, U gen){

}

template<>
void random_init<float, char> (float& x, char y){
    x=99;
}

