#include "leveldb_storage.hpp"

std::string LeveldbStorage::directory_add(const std::string& _path, const std::string& subdir)
{
    if(! boost::filesystem::is_directory(_path))
    {
        if(! boost::filesystem::create_directory(_path))
        {
            std::cerr << "Unable to create directory for database: " << _path << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return _path + "/" + subdir;
};

LeveldbStorage::LeveldbStorage(const std::string path, size_t default_ngram_length):
    path(path), default_ngram_length(default_ngram_length),
    fwd(directory_add(path, "fwd")),
    bwd(directory_add(path, "bwd"))
{
    // Set the terminals to Tries
    auto terminals = std::set<std::string>();
    terminals.insert(sentence_start);
    terminals.insert(sentence_end);
    fwd.set_terminals(terminals);
    bwd.set_terminals(terminals);
    // TODO: store default_ngram_length in a config DB
};


void LeveldbStorage::add_ngram(strVec& s, int freq)
{
    fwd.add_ngram(s, freq);
    bwd.add_ngram(reverse(s), freq);
};

void LeveldbStorage::add_sentence(std::vector<std::string> sentence, int freq, size_t ngram_length)
{
    if(! sentence.size())
        return;

    if(ngram_length == 0){
        ngram_length = default_ngram_length;
    }

    sentence.insert(sentence.begin(), sentence_start);
    sentence.push_back(sentence_end);

    for(auto it = sentence.begin(); it < sentence.end() - 1; it++)
    {
        fwd.add_ngram(std::vector<std::string>(it, std::min(it + ngram_length, sentence.end())), freq);
    }
    for(auto it = sentence.rbegin(); it < sentence.rend() - 1; it++)
    {
        bwd.add_ngram(std::vector<std::string>(it, std::min(it + ngram_length, sentence.rend())), freq);
    }
};

void LeveldbStorage::update_stats()
{
    fwd.update_stats();
    bwd.update_stats();
};

float LeveldbStorage::query_autonomy(strVec& ngram)
{
    float f = fwd.query_autonomy(ngram);
    if(isnan(f))
        return NAN;
    float b = bwd.query_autonomy(reverse(ngram));
    if(isnan(b))
        return NAN;
    return (f + b) / 2.f;
};

float LeveldbStorage::query_ev(strVec& ngram)
{
    float f = fwd.query_ev(ngram);
    if(isnan(f))
        return NAN;
    float b = bwd.query_ev(reverse(ngram));
    if(isnan(b))
        return NAN;
    return (f + b) / 2.f;
};

float LeveldbStorage::query_count(strVec& ngram)
{
    return (fwd.query_count(ngram) + bwd.query_count(reverse(ngram))) / 2.f;
};

float LeveldbStorage::query_entropy(strVec& ngram)
{
    float f = fwd.query_entropy(ngram);
    if(isnan(f))
        return NAN;
    float b = bwd.query_entropy(reverse(ngram));
    if(isnan(b))
        return NAN;
    return (f + b) / 2.f;
};

void LeveldbStorage::clear()
{
    fwd.clear();
    bwd.clear();
};

void LeveldbStorage::close()
{
    fwd.close();
    bwd.close();
};

