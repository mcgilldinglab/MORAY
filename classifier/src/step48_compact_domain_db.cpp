#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::unordered_map<std::string, std::string> values;
};

struct ReadRow {
    std::string read_id;
    std::string truth_domain;
    std::string error_id;
    std::string sequence;
    uint64_t row_index = 0;
};

struct ScoreRow {
    uint16_t total_valid_kmers = 0;
    uint16_t no_evidence_kmers = 0;
    uint16_t human_unique_kmers = 0;
    uint16_t bacteria_unique_kmers = 0;
    uint16_t fungi_unique_kmers = 0;
    uint16_t virus_unique_kmers = 0;
    std::string pred_domain;
    std::string pred_status;
};

struct QueryEvent {
    uint64_t kmer = 0;
    uint32_t read_index = 0;
};

std::string now_string() {
    std::time_t now = std::time(nullptr);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", std::localtime(&now));
    return std::string(buffer);
}

void log_msg(const std::string& message) {
    std::cerr << "[" << now_string() << "] " << message << std::endl;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected positional argument: " + key);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for argument: " + key);
        }
        args.values[key.substr(2)] = argv[++i];
    }
    return args;
}

std::string require_arg(const Args& args, const std::string& name) {
    auto it = args.values.find(name);
    if (it == args.values.end()) throw std::runtime_error("Missing required argument --" + name);
    return it->second;
}

bool has_arg(const Args& args, const std::string& name) {
    return args.values.find(name) != args.values.end();
}

std::string json_quote(const std::string& value) {
    std::ostringstream ss;
    ss << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') ss << '\\' << c;
        else if (c == '\n') ss << "\\n";
        else if (c == '\t') ss << "\\t";
        else ss << c;
    }
    ss << '"';
    return ss.str();
}

void write_json_summary(const fs::path& path, const std::vector<std::pair<std::string, std::string>>& fields) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not write summary: " + path.string());
    out << "{\n";
    for (size_t i = 0; i < fields.size(); ++i) {
        out << "  " << json_quote(fields[i].first) << ": " << fields[i].second;
        if (i + 1 != fields.size()) out << ',';
        out << "\n";
    }
    out << "}\n";
}

std::vector<std::string> split_tsv(const std::string& input) {
    std::string line = input;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    std::vector<std::string> out;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, '\t')) out.push_back(field);
    if (!line.empty() && line.back() == '\t') out.push_back("");
    return out;
}

std::unordered_map<std::string, size_t> header_index(const std::vector<std::string>& header) {
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < header.size(); ++i) index[header[i]] = i;
    return index;
}

std::string field_or_empty(
    const std::vector<std::string>& row,
    const std::unordered_map<std::string, size_t>& index,
    const std::string& name
) {
    auto it = index.find(name);
    if (it == index.end()) throw std::runtime_error("Missing required column: " + name);
    if (it->second >= row.size()) return "";
    return row[it->second];
}

// Like field_or_empty but returns "" when the column is absent instead of throwing. Used for
// truth_domain/error_id, which only matter for benchmark scoring -- a deployment reads TSV
// (just read_id + sequence, per prepare_reads.py) has neither and must still route.
std::string field_optional(
    const std::vector<std::string>& row,
    const std::unordered_map<std::string, size_t>& index,
    const std::string& name
) {
    auto it = index.find(name);
    if (it == index.end() || it->second >= row.size()) return "";
    return row[it->second];
}

std::string shard_name(int shard) {
    std::ostringstream ss;
    ss << "shard_" << std::setw(4) << std::setfill('0') << shard;
    return ss.str();
}

int base_bits(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1;
    }
}

class KmerEncoder {
   public:
    explicit KmerEncoder(int k_value) : k(k_value), shift(2 * (k_value - 1)) {
        if (k <= 0 || k > 31) throw std::runtime_error("This encoder requires 1 <= k <= 31");
        mask = (uint64_t{1} << (2 * k)) - 1;
    }

    template <typename Callback>
    uint16_t feed_sequence(const std::string& sequence, Callback cb) {
        uint64_t forward = 0;
        uint64_t reverse = 0;
        int valid = 0;
        uint16_t emitted = 0;
        for (char c : sequence) {
            int bits = base_bits(c);
            if (bits < 0) {
                forward = 0;
                reverse = 0;
                valid = 0;
                continue;
            }
            forward = ((forward << 2) | static_cast<uint64_t>(bits)) & mask;
            reverse = (reverse >> 2) | (static_cast<uint64_t>(3 - bits) << shift);
            if (valid < k) ++valid;
            if (valid >= k) {
                cb(std::min(forward, reverse));
                ++emitted;
            }
        }
        return emitted;
    }

   private:
    int k = 25;
    int shift = 48;
    uint64_t mask = 0;
};

std::vector<uint64_t> load_uint64_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open uint64 file: " + path.string());
    in.seekg(0, std::ios::end);
    std::streamoff bytes = in.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(uint64_t)) != 0) {
        throw std::runtime_error("Invalid uint64 file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint64_t> values(static_cast<size_t>(bytes / sizeof(uint64_t)));
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()), bytes);
        if (!in) throw std::runtime_error("Could not read uint64 file: " + path.string());
    }
    return values;
}

bool read_posting_header(std::ifstream& in, uint64_t& kmer, uint32_t& n_labels) {
    in.read(reinterpret_cast<char*>(&kmer), sizeof(uint64_t));
    if (!in) return false;
    in.read(reinterpret_cast<char*>(&n_labels), sizeof(uint32_t));
    if (!in) throw std::runtime_error("Truncated postings header");
    return true;
}

void append_posting_kmers(const fs::path& path, std::vector<uint64_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open postings shard: " + path.string());
    uint64_t kmer = 0;
    uint32_t n_labels = 0;
    while (read_posting_header(in, kmer, n_labels)) {
        out.push_back(kmer);
        in.seekg(static_cast<std::streamoff>(n_labels) * static_cast<std::streamoff>(sizeof(uint32_t)), std::ios::cur);
        if (!in) throw std::runtime_error("Could not skip postings labels: " + path.string());
    }
}

void sort_unique(std::vector<uint64_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::vector<uint64_t> load_combined_presence_shards(const fs::path& dir, int shard_id, int compact_shards, int source_shards) {
    std::vector<uint64_t> out;
    for (int shard = shard_id; shard < source_shards; shard += compact_shards) {
        fs::path path = dir / "shards" / (shard_name(shard) + ".kmers.bin");
        std::vector<uint64_t> values = load_uint64_file(path);
        out.insert(out.end(), values.begin(), values.end());
    }
    sort_unique(out);
    return out;
}

std::vector<uint64_t> load_combined_posting_shards(const fs::path& dir, int shard_id, int compact_shards, int source_shards) {
    std::vector<uint64_t> out;
    for (int shard = shard_id; shard < source_shards; shard += compact_shards) {
        append_posting_kmers(dir / "shards" / (shard_name(shard) + ".postings.bin"), out);
    }
    sort_unique(out);
    return out;
}

int popcount_u8(uint8_t value) {
    int out = 0;
    while (value) {
        out += value & 1;
        value >>= 1;
    }
    return out;
}

uint8_t mask_to_domain_id(uint8_t mask) {
    if (mask == 1) return 1;  // human
    if (mask == 2) return 2;  // bacteria
    if (mask == 4) return 3;  // fungi
    if (mask == 8) return 4;  // virus
    return 0;
}

std::string domain_id_to_name(uint8_t domain_id) {
    if (domain_id == 1) return "human";
    if (domain_id == 2) return "bacteria";
    if (domain_id == 3) return "fungi";
    if (domain_id == 4) return "virus";
    return "";
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void flush_buffer(std::ofstream& out, std::vector<uint64_t>& buffer) {
    if (buffer.empty()) return;
    out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(uint64_t)));
    if (!out) throw std::runtime_error("Could not write compact buffer");
    buffer.clear();
}

void mode_build_shard(const Args& args) {
    const auto start = std::chrono::steady_clock::now();
    fs::path output_dir(require_arg(args, "output-dir"));
    int shard_id = std::stoi(require_arg(args, "shard-id"));
    int compact_shards = std::stoi(require_arg(args, "compact-shards"));
    int source_shards = std::stoi(require_arg(args, "source-shards"));
    int human_shards = std::stoi(require_arg(args, "human-shards"));
    fs::path human_index_dir(require_arg(args, "human-index-dir"));
    fs::path bacteria_index_dir(require_arg(args, "bacteria-index-dir"));
    fs::path fungi_presence_dir(require_arg(args, "fungi-presence-dir"));
    fs::path virus_index_dir(require_arg(args, "virus-index-dir"));
    size_t buffer_records = has_arg(args, "buffer-records") ? static_cast<size_t>(std::stoull(require_arg(args, "buffer-records"))) : 1048576;

    fs::path shard_dir = output_dir / "shards";
    fs::create_directories(shard_dir);
    fs::path out_path = shard_dir / (shard_name(shard_id) + ".compact.bin");
    fs::path done_path = shard_dir / (shard_name(shard_id) + ".done.json");
    if (fs::exists(done_path) && fs::exists(out_path) && !has_arg(args, "force")) {
        log_msg("Compact shard already completed, skipping: " + done_path.string());
        return;
    }

    if (compact_shards != human_shards) {
        throw std::runtime_error("This build mode expects compact_shards == human_shards");
    }
    log_msg("Loading domain presence for compact shard " + std::to_string(shard_id));
    std::vector<uint64_t> human = load_uint64_file(human_index_dir / "shards" / (shard_name(shard_id) + ".kmers.bin"));
    std::vector<uint64_t> bacteria = load_combined_posting_shards(bacteria_index_dir, shard_id, compact_shards, source_shards);
    std::vector<uint64_t> fungi = load_combined_presence_shards(fungi_presence_dir, shard_id, compact_shards, source_shards);
    std::vector<uint64_t> virus = load_combined_posting_shards(virus_index_dir, shard_id, compact_shards, source_shards);

    fs::path tmp_path = out_path;
    tmp_path += ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not write compact shard: " + tmp_path.string());
    std::vector<uint64_t> buffer;
    buffer.reserve(buffer_records);

    size_t ih = 0, ib = 0, iff = 0, iv = 0;
    uint64_t total_union = 0;
    uint64_t shared_or_multi = 0;
    uint64_t unique_human = 0;
    uint64_t unique_bacteria = 0;
    uint64_t unique_fungi = 0;
    uint64_t unique_virus = 0;

    while (ih < human.size() || ib < bacteria.size() || iff < fungi.size() || iv < virus.size()) {
        uint64_t kmer = std::numeric_limits<uint64_t>::max();
        if (ih < human.size()) kmer = std::min(kmer, human[ih]);
        if (ib < bacteria.size()) kmer = std::min(kmer, bacteria[ib]);
        if (iff < fungi.size()) kmer = std::min(kmer, fungi[iff]);
        if (iv < virus.size()) kmer = std::min(kmer, virus[iv]);
        uint8_t mask = 0;
        if (ih < human.size() && human[ih] == kmer) {
            mask |= 1;
            while (ih < human.size() && human[ih] == kmer) ++ih;
        }
        if (ib < bacteria.size() && bacteria[ib] == kmer) {
            mask |= 2;
            while (ib < bacteria.size() && bacteria[ib] == kmer) ++ib;
        }
        if (iff < fungi.size() && fungi[iff] == kmer) {
            mask |= 4;
            while (iff < fungi.size() && fungi[iff] == kmer) ++iff;
        }
        if (iv < virus.size() && virus[iv] == kmer) {
            mask |= 8;
            while (iv < virus.size() && virus[iv] == kmer) ++iv;
        }
        ++total_union;
        uint8_t domain_id = mask_to_domain_id(mask);
        if (domain_id == 0) {
            ++shared_or_multi;
            continue;
        }
        if (domain_id == 1) ++unique_human;
        else if (domain_id == 2) ++unique_bacteria;
        else if (domain_id == 3) ++unique_fungi;
        else if (domain_id == 4) ++unique_virus;
        buffer.push_back((kmer << 3) | static_cast<uint64_t>(domain_id));
        if (buffer.size() >= buffer_records) flush_buffer(out, buffer);
    }
    flush_buffer(out, buffer);
    out.close();
    fs::rename(tmp_path, out_path);

    const auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    uint64_t compact_records = unique_human + unique_bacteria + unique_fungi + unique_virus;
    write_json_summary(done_path, {
        {"created_at", json_quote(now_string())},
        {"status", json_quote("completed")},
        {"mode", json_quote("build-compact-domain-shard")},
        {"shard_id", std::to_string(shard_id)},
        {"compact_shards", std::to_string(compact_shards)},
        {"source_shards", std::to_string(source_shards)},
        {"human_input_kmers", std::to_string(human.size())},
        {"bacteria_input_kmers", std::to_string(bacteria.size())},
        {"fungi_input_kmers", std::to_string(fungi.size())},
        {"virus_input_kmers", std::to_string(virus.size())},
        {"total_union_kmers", std::to_string(total_union)},
        {"shared_or_multi_domain_kmers_discarded", std::to_string(shared_or_multi)},
        {"compact_records", std::to_string(compact_records)},
        {"human_unique_kmers", std::to_string(unique_human)},
        {"bacteria_unique_kmers", std::to_string(unique_bacteria)},
        {"fungi_unique_kmers", std::to_string(unique_fungi)},
        {"virus_unique_kmers", std::to_string(unique_virus)},
        {"output_path", json_quote(out_path.string())},
        {"output_bytes", std::to_string(fs::file_size(out_path))},
        {"elapsed_seconds", std::to_string(elapsed)}
    });
    log_msg("Completed compact shard " + std::to_string(shard_id) + " records=" + std::to_string(compact_records));
}

std::vector<ReadRow> load_reads(const fs::path& reads_tsv, uint64_t max_reads) {
    std::ifstream in(reads_tsv);
    if (!in) throw std::runtime_error("Could not open reads TSV: " + reads_tsv.string());
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Empty reads TSV: " + reads_tsv.string());
    auto index = header_index(split_tsv(line));
    std::vector<ReadRow> reads;
    uint64_t row_index = 0;
    while (std::getline(in, line)) {
        if (max_reads > 0 && reads.size() >= max_reads) break;
        auto row = split_tsv(line);
        ReadRow read;
        read.row_index = row_index;
        read.read_id = field_or_empty(row, index, "read_id");
        read.truth_domain = field_optional(row, index, "truth_domain");
        read.error_id = field_optional(row, index, "error_id");
        read.sequence = field_or_empty(row, index, "sequence");
        reads.push_back(std::move(read));
        ++row_index;
    }
    return reads;
}

class MMapShard {
   public:
    MMapShard() = default;
    MMapShard(const MMapShard&) = delete;
    MMapShard& operator=(const MMapShard&) = delete;
    MMapShard(MMapShard&& other) noexcept {
        *this = std::move(other);
    }
    MMapShard& operator=(MMapShard&& other) noexcept {
        if (this != &other) {
            close();
            fd = other.fd;
            data = other.data;
            bytes = other.bytes;
            count = other.count;
            other.fd = -1;
            other.data = nullptr;
            other.bytes = 0;
            other.count = 0;
        }
        return *this;
    }
    ~MMapShard() { close(); }

    void open_file(const fs::path& path) {
        close();
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Could not open compact shard: " + path.string());
        struct stat st {};
        if (fstat(fd, &st) != 0) throw std::runtime_error("Could not stat compact shard: " + path.string());
        bytes = static_cast<size_t>(st.st_size);
        if (bytes % sizeof(uint64_t) != 0) throw std::runtime_error("Invalid compact shard byte size: " + path.string());
        count = bytes / sizeof(uint64_t);
        if (bytes == 0) {
            data = nullptr;
            return;
        }
        void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("Could not mmap compact shard: " + path.string());
        data = static_cast<const uint64_t*>(mapped);
    }

    uint8_t lookup(uint64_t kmer) const {
        if (count == 0) return 0;
        uint64_t key = kmer << 3;
        const uint64_t* begin = data;
        const uint64_t* end = data + count;
        const uint64_t* it = std::lower_bound(begin, end, key);
        if (it == end || ((*it) >> 3) != kmer) return 0;
        return static_cast<uint8_t>((*it) & 7);
    }

    void close() {
        if (data && bytes > 0) {
            munmap(const_cast<uint64_t*>(data), bytes);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        fd = -1;
        data = nullptr;
        bytes = 0;
        count = 0;
    }

    size_t size_bytes() const { return bytes; }
    size_t records() const { return count; }
    uint64_t at(size_t index) const { return data[index]; }

   private:
    int fd = -1;
    const uint64_t* data = nullptr;
    size_t bytes = 0;
    size_t count = 0;
};

class MMapShard6 {
   public:
    MMapShard6() = default;
    MMapShard6(const MMapShard6&) = delete;
    MMapShard6& operator=(const MMapShard6&) = delete;
    MMapShard6(MMapShard6&& other) noexcept {
        *this = std::move(other);
    }
    MMapShard6& operator=(MMapShard6&& other) noexcept {
        if (this != &other) {
            close();
            fd = other.fd;
            data = other.data;
            bytes = other.bytes;
            count = other.count;
            other.fd = -1;
            other.data = nullptr;
            other.bytes = 0;
            other.count = 0;
        }
        return *this;
    }
    ~MMapShard6() { close(); }

    void open_file(const fs::path& path) {
        close();
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Could not open compact6 shard: " + path.string());
        struct stat st {};
        if (fstat(fd, &st) != 0) throw std::runtime_error("Could not stat compact6 shard: " + path.string());
        bytes = static_cast<size_t>(st.st_size);
        if (bytes % 6 != 0) throw std::runtime_error("Invalid compact6 shard byte size: " + path.string());
        count = bytes / 6;
        if (bytes == 0) {
            data = nullptr;
            return;
        }
        void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) throw std::runtime_error("Could not mmap compact6 shard: " + path.string());
        data = static_cast<const uint8_t*>(mapped);
    }

    uint64_t at(size_t index) const {
        const uint8_t* p = data + index * 6;
        return static_cast<uint64_t>(p[0]) |
               (static_cast<uint64_t>(p[1]) << 8) |
               (static_cast<uint64_t>(p[2]) << 16) |
               (static_cast<uint64_t>(p[3]) << 24) |
               (static_cast<uint64_t>(p[4]) << 32) |
               (static_cast<uint64_t>(p[5]) << 40);
    }

    void close() {
        if (data && bytes > 0) {
            munmap(const_cast<uint8_t*>(data), bytes);
        }
        if (fd >= 0) {
            ::close(fd);
        }
        fd = -1;
        data = nullptr;
        bytes = 0;
        count = 0;
    }

    size_t size_bytes() const { return bytes; }
    size_t records() const { return count; }

   private:
    int fd = -1;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    size_t count = 0;
};

void write_compact6_record(std::ofstream& out, uint64_t packed6) {
    uint8_t bytes[6] = {
        static_cast<uint8_t>(packed6 & 0xff),
        static_cast<uint8_t>((packed6 >> 8) & 0xff),
        static_cast<uint8_t>((packed6 >> 16) & 0xff),
        static_cast<uint8_t>((packed6 >> 24) & 0xff),
        static_cast<uint8_t>((packed6 >> 32) & 0xff),
        static_cast<uint8_t>((packed6 >> 40) & 0xff),
    };
    out.write(reinterpret_cast<const char*>(bytes), 6);
    if (!out) throw std::runtime_error("Could not write compact6 record");
}

bool keep_bacteria_record(uint64_t kmer, uint32_t retain_per_million) {
    if (retain_per_million >= 1000000U) return true;
    if (retain_per_million == 0U) return false;
    return static_cast<uint32_t>(splitmix64(kmer) % 1000000ULL) < retain_per_million;
}

void mode_convert_compact6_shard(const Args& args) {
    const auto start = std::chrono::steady_clock::now();
    fs::path input_dir(require_arg(args, "input-dir"));
    fs::path output_dir(require_arg(args, "output-dir"));
    int shard_id = std::stoi(require_arg(args, "shard-id"));
    uint64_t compact_shards = static_cast<uint64_t>(std::stoull(require_arg(args, "compact-shards")));
    uint32_t bacteria_retain_per_million = has_arg(args, "bacteria-retain-per-million")
        ? static_cast<uint32_t>(std::stoul(require_arg(args, "bacteria-retain-per-million")))
        : 1000000U;
    std::string db_label = has_arg(args, "db-label") ? require_arg(args, "db-label") : "compact6";
    if (bacteria_retain_per_million > 1000000U) {
        throw std::runtime_error("--bacteria-retain-per-million must be between 0 and 1000000");
    }

    fs::path shard_dir = output_dir / "shards";
    fs::create_directories(shard_dir);
    fs::path in_path = input_dir / "shards" / (shard_name(shard_id) + ".compact.bin");
    fs::path out_path = shard_dir / (shard_name(shard_id) + ".compact6.bin");
    fs::path done_path = shard_dir / (shard_name(shard_id) + ".done.json");
    if (fs::exists(done_path) && fs::exists(out_path) && !has_arg(args, "force")) {
        log_msg("Compact6 shard already completed, skipping: " + done_path.string());
        return;
    }

    MMapShard in;
    in.open_file(in_path);
    fs::path tmp_path = out_path;
    tmp_path += ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not write compact6 shard: " + tmp_path.string());

    uint64_t input_records = in.records();
    uint64_t output_records = 0;
    uint64_t retained_human = 0;
    uint64_t retained_bacteria = 0;
    uint64_t retained_fungi = 0;
    uint64_t retained_virus = 0;
    uint64_t dropped_bacteria = 0;

    for (size_t i = 0; i < in.records(); ++i) {
        uint64_t packed = in.at(i);
        uint64_t kmer = packed >> 3;
        uint8_t domain_id = static_cast<uint8_t>(packed & 7);
        if (domain_id == 2 && !keep_bacteria_record(kmer, bacteria_retain_per_million)) {
            ++dropped_bacteria;
            continue;
        }
        if (kmer % compact_shards != static_cast<uint64_t>(shard_id)) {
            throw std::runtime_error("Input compact shard contains unexpected kmer modulo for shard " + std::to_string(shard_id));
        }
        uint64_t quotient = kmer / compact_shards;
        uint64_t packed6 = (quotient << 3) | static_cast<uint64_t>(domain_id);
        if (packed6 >= (1ULL << 48)) {
            throw std::runtime_error("compact6 encoded value does not fit in 48 bits");
        }
        write_compact6_record(out, packed6);
        ++output_records;
        if (domain_id == 1) ++retained_human;
        else if (domain_id == 2) ++retained_bacteria;
        else if (domain_id == 3) ++retained_fungi;
        else if (domain_id == 4) ++retained_virus;
    }
    out.close();
    fs::rename(tmp_path, out_path);

    const auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    write_json_summary(done_path, {
        {"created_at", json_quote(now_string())},
        {"status", json_quote("completed")},
        {"mode", json_quote("convert-compact-domain-shard-to-compact6")},
        {"db_label", json_quote(db_label)},
        {"shard_id", std::to_string(shard_id)},
        {"compact_shards", std::to_string(compact_shards)},
        {"input_path", json_quote(in_path.string())},
        {"output_path", json_quote(out_path.string())},
        {"input_records", std::to_string(input_records)},
        {"output_records", std::to_string(output_records)},
        {"bacteria_retain_per_million", std::to_string(bacteria_retain_per_million)},
        {"retained_human_kmers", std::to_string(retained_human)},
        {"retained_bacteria_kmers", std::to_string(retained_bacteria)},
        {"retained_fungi_kmers", std::to_string(retained_fungi)},
        {"retained_virus_kmers", std::to_string(retained_virus)},
        {"dropped_bacteria_kmers", std::to_string(dropped_bacteria)},
        {"output_bytes", std::to_string(fs::file_size(out_path))},
        {"elapsed_seconds", std::to_string(elapsed)}
    });
    log_msg("Completed compact6 shard " + std::to_string(shard_id) + " label=" + db_label + " records=" + std::to_string(output_records));
}

void finalize_score(ScoreRow& score) {
    uint16_t vals[4] = {
        score.human_unique_kmers,
        score.bacteria_unique_kmers,
        score.fungi_unique_kmers,
        score.virus_unique_kmers,
    };
    uint16_t best = 0;
    int best_i = -1;
    int ties = 0;
    for (int i = 0; i < 4; ++i) {
        if (vals[i] > best) {
            best = vals[i];
            best_i = i;
            ties = 1;
        } else if (vals[i] == best && best > 0) {
            ++ties;
        }
    }
    if (best == 0) {
        score.pred_domain = "no_call";
        score.pred_status = "no_call";
    } else if (ties > 1) {
        score.pred_domain = "ambiguous";
        score.pred_status = "ambiguous";
    } else {
        score.pred_domain = domain_id_to_name(static_cast<uint8_t>(best_i + 1));
        score.pred_status = "called";
    }
}

fs::path default_partial_dir_for_output(const fs::path& output_tsv) {
    return output_tsv.parent_path() / (output_tsv.stem().string() + ".partials");
}

bool partial_done_exists(const fs::path& partial_dir, int shard) {
    fs::path bin_path = partial_dir / (shard_name(shard) + ".partial.bin");
    fs::path done_path = partial_dir / (shard_name(shard) + ".done.json");
    return fs::exists(bin_path) && fs::exists(done_path);
}

void write_uint16(std::ofstream& out, uint16_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(uint16_t));
}

void write_uint32(std::ofstream& out, uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(uint32_t));
}

void add_partial_counts(const fs::path& bin_path, std::vector<ScoreRow>& scores) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open partial counts: " + bin_path.string());
    while (true) {
        uint32_t read_index = 0;
        uint16_t human = 0;
        uint16_t bacteria = 0;
        uint16_t fungi = 0;
        uint16_t virus = 0;
        in.read(reinterpret_cast<char*>(&read_index), sizeof(uint32_t));
        if (!in) {
            if (in.eof()) break;
            throw std::runtime_error("Could not read partial read index: " + bin_path.string());
        }
        in.read(reinterpret_cast<char*>(&human), sizeof(uint16_t));
        in.read(reinterpret_cast<char*>(&bacteria), sizeof(uint16_t));
        in.read(reinterpret_cast<char*>(&fungi), sizeof(uint16_t));
        in.read(reinterpret_cast<char*>(&virus), sizeof(uint16_t));
        if (!in) throw std::runtime_error("Truncated partial counts: " + bin_path.string());
        if (read_index >= scores.size()) throw std::runtime_error("Partial read index out of range: " + bin_path.string());
        scores[read_index].human_unique_kmers += human;
        scores[read_index].bacteria_unique_kmers += bacteria;
        scores[read_index].fungi_unique_kmers += fungi;
        scores[read_index].virus_unique_kmers += virus;
    }
}

void write_partial_counts(
    const fs::path& partial_dir,
    int shard,
    const std::vector<std::array<uint16_t, 4>>& counts,
    uint64_t events,
    uint64_t compact_records,
    double elapsed
) {
    fs::create_directories(partial_dir);
    fs::path bin_path = partial_dir / (shard_name(shard) + ".partial.bin");
    fs::path done_path = partial_dir / (shard_name(shard) + ".done.json");
    fs::path tmp_path = bin_path;
    tmp_path += ".tmp";
    std::ofstream out(tmp_path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not write partial counts: " + tmp_path.string());
    uint64_t nonzero_reads = 0;
    for (uint32_t read_index = 0; read_index < counts.size(); ++read_index) {
        const auto& c = counts[read_index];
        if (c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0) continue;
        write_uint32(out, read_index);
        write_uint16(out, c[0]);
        write_uint16(out, c[1]);
        write_uint16(out, c[2]);
        write_uint16(out, c[3]);
        ++nonzero_reads;
    }
    out.close();
    fs::rename(tmp_path, bin_path);
    write_json_summary(done_path, {
        {"created_at", json_quote(now_string())},
        {"status", json_quote("completed")},
        {"mode", json_quote("query-compact-domain-db-stream-partial-shard")},
        {"shard_id", std::to_string(shard)},
        {"events", std::to_string(events)},
        {"compact_records", std::to_string(compact_records)},
        {"nonzero_reads", std::to_string(nonzero_reads)},
        {"output_path", json_quote(bin_path.string())},
        {"output_bytes", std::to_string(fs::file_size(bin_path))},
        {"elapsed_seconds", std::to_string(elapsed)}
    });
}

void process_stream_shard(
    int shard,
    const fs::path& compact_dir,
    const fs::path& partial_dir,
    std::vector<QueryEvent>& events,
    size_t read_count
) {
    const auto start = std::chrono::steady_clock::now();
    if (partial_done_exists(partial_dir, shard)) return;
    std::sort(events.begin(), events.end(), [](const QueryEvent& a, const QueryEvent& b) {
        if (a.kmer != b.kmer) return a.kmer < b.kmer;
        return a.read_index < b.read_index;
    });

    MMapShard db;
    db.open_file(compact_dir / "shards" / (shard_name(shard) + ".compact.bin"));
    std::vector<std::array<uint16_t, 4>> counts(read_count);

    size_t rec = 0;
    size_t event = 0;
    while (event < events.size()) {
        uint64_t query_kmer = events[event].kmer;
        while (rec < db.records() && (db.at(rec) >> 3) < query_kmer) ++rec;
        uint8_t domain_id = 0;
        if (rec < db.records() && (db.at(rec) >> 3) == query_kmer) {
            domain_id = static_cast<uint8_t>(db.at(rec) & 7);
        }
        while (event < events.size() && events[event].kmer == query_kmer) {
            if (domain_id >= 1 && domain_id <= 4) {
                ++counts[events[event].read_index][domain_id - 1];
            }
            ++event;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    write_partial_counts(partial_dir, shard, counts, events.size(), db.records(), elapsed);
    events.clear();
    events.shrink_to_fit();
}

void process_stream_shard6(
    int shard,
    int compact_shards,
    const fs::path& compact_dir,
    const fs::path& partial_dir,
    std::vector<QueryEvent>& events,
    size_t read_count
) {
    const auto start = std::chrono::steady_clock::now();
    if (partial_done_exists(partial_dir, shard)) return;
    std::sort(events.begin(), events.end(), [](const QueryEvent& a, const QueryEvent& b) {
        if (a.kmer != b.kmer) return a.kmer < b.kmer;
        return a.read_index < b.read_index;
    });

    MMapShard6 db;
    db.open_file(compact_dir / "shards" / (shard_name(shard) + ".compact6.bin"));
    std::vector<std::array<uint16_t, 4>> counts(read_count);

    size_t rec = 0;
    size_t event = 0;
    while (event < events.size()) {
        uint64_t query_kmer = events[event].kmer;
        while (rec < db.records()) {
            uint64_t packed6 = db.at(rec);
            uint64_t db_kmer = (packed6 >> 3) * static_cast<uint64_t>(compact_shards) + static_cast<uint64_t>(shard);
            if (db_kmer >= query_kmer) break;
            ++rec;
        }
        uint8_t domain_id = 0;
        if (rec < db.records()) {
            uint64_t packed6 = db.at(rec);
            uint64_t db_kmer = (packed6 >> 3) * static_cast<uint64_t>(compact_shards) + static_cast<uint64_t>(shard);
            if (db_kmer == query_kmer) {
                domain_id = static_cast<uint8_t>(packed6 & 7);
            }
        }
        while (event < events.size() && events[event].kmer == query_kmer) {
            if (domain_id >= 1 && domain_id <= 4) {
                ++counts[events[event].read_index][domain_id - 1];
            }
            ++event;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    write_partial_counts(partial_dir, shard, counts, events.size(), db.records(), elapsed);
    events.clear();
    events.shrink_to_fit();
}

void mode_query_reads_stream(const Args& args) {
    const auto start = std::chrono::steady_clock::now();
    fs::path reads_tsv(require_arg(args, "reads-tsv"));
    fs::path compact_dir(require_arg(args, "compact-dir"));
    fs::path output_tsv(require_arg(args, "output-tsv"));
    int compact_shards = std::stoi(require_arg(args, "compact-shards"));
    int threads = std::stoi(require_arg(args, "threads"));
    int k = has_arg(args, "k") ? std::stoi(require_arg(args, "k")) : 25;
    uint64_t max_reads = has_arg(args, "max-reads") ? std::stoull(require_arg(args, "max-reads")) : 0;
    fs::path partial_dir = has_arg(args, "partial-dir") ? fs::path(require_arg(args, "partial-dir")) : default_partial_dir_for_output(output_tsv);
    fs::path done_path = output_tsv;
    done_path += ".done.json";
    if (fs::exists(done_path) && fs::exists(output_tsv) && !has_arg(args, "force")) {
        log_msg("Compact streaming query already completed, skipping: " + done_path.string());
        return;
    }

    std::vector<ReadRow> reads = load_reads(reads_tsv, max_reads);
    log_msg("Loaded reads=" + std::to_string(reads.size()));
    fs::create_directories(partial_dir);

    std::vector<bool> completed_shards(static_cast<size_t>(compact_shards), false);
    uint64_t completed_before_start = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        if (partial_done_exists(partial_dir, shard)) {
            completed_shards[static_cast<size_t>(shard)] = true;
            ++completed_before_start;
        }
    }
    log_msg("Existing completed partial shards=" + std::to_string(completed_before_start));

    const auto event_start = std::chrono::steady_clock::now();
    std::vector<std::vector<QueryEvent>> events(static_cast<size_t>(compact_shards));
    std::vector<ScoreRow> scores(reads.size());
    KmerEncoder encoder(k);
    uint64_t raw_kmers = 0;
    for (size_t i = 0; i < reads.size(); ++i) {
        uint16_t emitted = encoder.feed_sequence(reads[i].sequence, [&](uint64_t kmer) {
            int shard = static_cast<int>(kmer % static_cast<uint64_t>(compact_shards));
            if (!completed_shards[static_cast<size_t>(shard)]) {
                events[static_cast<size_t>(shard)].push_back({kmer, static_cast<uint32_t>(i)});
            }
        });
        scores[i].total_valid_kmers = emitted;
        raw_kmers += emitted;
    }
    const auto event_end = std::chrono::steady_clock::now();
    double event_elapsed = std::chrono::duration<double>(event_end - event_start).count();

    uint64_t pending_events = 0;
    uint64_t pending_shards = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        pending_events += events[static_cast<size_t>(shard)].size();
        if (!completed_shards[static_cast<size_t>(shard)]) ++pending_shards;
    }
    log_msg("Prepared pending shards=" + std::to_string(pending_shards) + " pending_events=" + std::to_string(pending_events));

    const auto shard_start = std::chrono::steady_clock::now();
    std::atomic<int> next_shard{0};
    std::atomic<int> processed_now{0};
    std::atomic<bool> failed{false};
    std::string first_error;
    std::mutex error_mutex;
    auto worker = [&]() {
        while (!failed.load()) {
            int shard = next_shard.fetch_add(1);
            if (shard >= compact_shards) break;
            if (completed_shards[static_cast<size_t>(shard)]) continue;
            try {
                process_stream_shard(shard, compact_dir, partial_dir, events[static_cast<size_t>(shard)], reads.size());
                int done_now = processed_now.fetch_add(1) + 1;
                log_msg("Completed stream shard " + std::to_string(shard) + " (" + std::to_string(done_now) + "/" + std::to_string(pending_shards) + ")");
            } catch (const std::exception& exc) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) first_error = exc.what();
            }
        }
    };
    std::vector<std::thread> workers;
    int actual_threads = std::max(1, threads);
    workers.reserve(static_cast<size_t>(actual_threads));
    for (int t = 0; t < actual_threads; ++t) workers.emplace_back(worker);
    for (auto& t : workers) t.join();
    if (failed.load()) throw std::runtime_error("Streaming shard worker failed: " + first_error);
    const auto shard_end = std::chrono::steady_clock::now();
    double shard_elapsed = std::chrono::duration<double>(shard_end - shard_start).count();

    const auto aggregate_start = std::chrono::steady_clock::now();
    uint64_t partial_shards = 0;
    uint64_t db_records = 0;
    uint64_t db_bytes = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        fs::path partial_path = partial_dir / (shard_name(shard) + ".partial.bin");
        fs::path done_json = partial_dir / (shard_name(shard) + ".done.json");
        fs::path compact_path = compact_dir / "shards" / (shard_name(shard) + ".compact.bin");
        if (!fs::exists(partial_path) || !fs::exists(done_json)) {
            throw std::runtime_error("Missing completed partial shard: " + shard_name(shard));
        }
        add_partial_counts(partial_path, scores);
        ++partial_shards;
        db_bytes += fs::file_size(compact_path);
        db_records += fs::file_size(compact_path) / sizeof(uint64_t);
    }
    for (auto& score : scores) {
        uint16_t called = score.human_unique_kmers + score.bacteria_unique_kmers + score.fungi_unique_kmers + score.virus_unique_kmers;
        score.no_evidence_kmers = score.total_valid_kmers >= called ? score.total_valid_kmers - called : 0;
        finalize_score(score);
    }
    const auto aggregate_end = std::chrono::steady_clock::now();
    double aggregate_elapsed = std::chrono::duration<double>(aggregate_end - aggregate_start).count();

    const auto write_start = std::chrono::steady_clock::now();
    fs::create_directories(output_tsv.parent_path());
    fs::path tmp_path = output_tsv;
    tmp_path += ".tmp";
    std::ofstream out(tmp_path);
    if (!out) throw std::runtime_error("Could not write compact query TSV: " + tmp_path.string());
    out << "read_id\trow_index\terror_id\ttruth_domain\ttotal_valid_kmers\tno_evidence_kmers\t"
        << "human_unique_kmers\tbacteria_unique_kmers\tfungi_unique_kmers\tvirus_unique_kmers\tpred_domain\tpred_status\n";
    for (size_t i = 0; i < reads.size(); ++i) {
        const auto& r = reads[i];
        const auto& s = scores[i];
        out << r.read_id << '\t' << r.row_index << '\t' << r.error_id << '\t' << r.truth_domain << '\t'
            << s.total_valid_kmers << '\t' << s.no_evidence_kmers << '\t'
            << s.human_unique_kmers << '\t' << s.bacteria_unique_kmers << '\t'
            << s.fungi_unique_kmers << '\t' << s.virus_unique_kmers << '\t'
            << s.pred_domain << '\t' << s.pred_status << '\n';
    }
    out.close();
    fs::rename(tmp_path, output_tsv);
    const auto write_end = std::chrono::steady_clock::now();
    double write_elapsed = std::chrono::duration<double>(write_end - write_start).count();

    const auto end = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(end - start).count();
    double query_elapsed = event_elapsed + shard_elapsed + aggregate_elapsed;
    write_json_summary(done_path, {
        {"created_at", json_quote(now_string())},
        {"status", json_quote("completed")},
        {"mode", json_quote("query-compact-domain-db-stream")},
        {"reads_tsv", json_quote(reads_tsv.string())},
        {"compact_dir", json_quote(compact_dir.string())},
        {"partial_dir", json_quote(partial_dir.string())},
        {"output_tsv", json_quote(output_tsv.string())},
        {"compact_shards", std::to_string(compact_shards)},
        {"threads", std::to_string(actual_threads)},
        {"reads", std::to_string(reads.size())},
        {"raw_kmers", std::to_string(raw_kmers)},
        {"partial_shards", std::to_string(partial_shards)},
        {"partial_shards_completed_before_start", std::to_string(completed_before_start)},
        {"db_records", std::to_string(db_records)},
        {"db_bytes", std::to_string(db_bytes)},
        {"event_generation_elapsed_seconds", std::to_string(event_elapsed)},
        {"shard_processing_elapsed_seconds", std::to_string(shard_elapsed)},
        {"aggregation_elapsed_seconds", std::to_string(aggregate_elapsed)},
        {"output_write_elapsed_seconds", std::to_string(write_elapsed)},
        {"query_elapsed_seconds", std::to_string(query_elapsed)},
        {"total_elapsed_seconds", std::to_string(total_elapsed)},
        {"reads_per_second_query_only", std::to_string(reads.empty() ? 0.0 : static_cast<double>(reads.size()) / query_elapsed)},
        {"kmer_lookups_per_second_query_only", std::to_string(raw_kmers == 0 ? 0.0 : static_cast<double>(raw_kmers) / query_elapsed)}
    });
    log_msg("Completed compact streaming query reads=" + std::to_string(reads.size()) + " query_elapsed=" + std::to_string(query_elapsed));
}

void mode_query_reads_stream6(const Args& args) {
    const auto start = std::chrono::steady_clock::now();
    fs::path reads_tsv(require_arg(args, "reads-tsv"));
    fs::path compact_dir(require_arg(args, "compact-dir"));
    fs::path output_tsv(require_arg(args, "output-tsv"));
    int compact_shards = std::stoi(require_arg(args, "compact-shards"));
    int threads = std::stoi(require_arg(args, "threads"));
    int k = has_arg(args, "k") ? std::stoi(require_arg(args, "k")) : 25;
    uint64_t max_reads = has_arg(args, "max-reads") ? std::stoull(require_arg(args, "max-reads")) : 0;
    fs::path partial_dir = has_arg(args, "partial-dir") ? fs::path(require_arg(args, "partial-dir")) : default_partial_dir_for_output(output_tsv);
    std::string method_label = has_arg(args, "method-label") ? require_arg(args, "method-label") : "compact_domain_k25_compact6";
    fs::path done_path = output_tsv;
    done_path += ".done.json";
    if (fs::exists(done_path) && fs::exists(output_tsv) && !has_arg(args, "force")) {
        log_msg("Compact6 streaming query already completed, skipping: " + done_path.string());
        return;
    }

    std::vector<ReadRow> reads = load_reads(reads_tsv, max_reads);
    log_msg("Loaded reads=" + std::to_string(reads.size()));
    fs::create_directories(partial_dir);

    std::vector<bool> completed_shards(static_cast<size_t>(compact_shards), false);
    uint64_t completed_before_start = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        if (partial_done_exists(partial_dir, shard)) {
            completed_shards[static_cast<size_t>(shard)] = true;
            ++completed_before_start;
        }
    }
    log_msg("Existing completed partial shards=" + std::to_string(completed_before_start));

    const auto event_start = std::chrono::steady_clock::now();
    std::vector<std::vector<QueryEvent>> events(static_cast<size_t>(compact_shards));
    std::vector<ScoreRow> scores(reads.size());
    KmerEncoder encoder(k);
    uint64_t raw_kmers = 0;
    for (size_t i = 0; i < reads.size(); ++i) {
        uint16_t emitted = encoder.feed_sequence(reads[i].sequence, [&](uint64_t kmer) {
            int shard = static_cast<int>(kmer % static_cast<uint64_t>(compact_shards));
            if (!completed_shards[static_cast<size_t>(shard)]) {
                events[static_cast<size_t>(shard)].push_back({kmer, static_cast<uint32_t>(i)});
            }
        });
        scores[i].total_valid_kmers = emitted;
        raw_kmers += emitted;
    }
    const auto event_end = std::chrono::steady_clock::now();
    double event_elapsed = std::chrono::duration<double>(event_end - event_start).count();

    uint64_t pending_events = 0;
    uint64_t pending_shards = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        pending_events += events[static_cast<size_t>(shard)].size();
        if (!completed_shards[static_cast<size_t>(shard)]) ++pending_shards;
    }
    log_msg("Prepared compact6 pending shards=" + std::to_string(pending_shards) + " pending_events=" + std::to_string(pending_events));

    const auto shard_start = std::chrono::steady_clock::now();
    std::atomic<int> next_shard{0};
    std::atomic<int> processed_now{0};
    std::atomic<bool> failed{false};
    std::string first_error;
    std::mutex error_mutex;
    auto worker = [&]() {
        while (!failed.load()) {
            int shard = next_shard.fetch_add(1);
            if (shard >= compact_shards) break;
            if (completed_shards[static_cast<size_t>(shard)]) continue;
            try {
                process_stream_shard6(shard, compact_shards, compact_dir, partial_dir, events[static_cast<size_t>(shard)], reads.size());
                int done_now = processed_now.fetch_add(1) + 1;
                log_msg("Completed compact6 stream shard " + std::to_string(shard) + " (" + std::to_string(done_now) + "/" + std::to_string(pending_shards) + ")");
            } catch (const std::exception& exc) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) first_error = exc.what();
            }
        }
    };
    std::vector<std::thread> workers;
    int actual_threads = std::max(1, threads);
    workers.reserve(static_cast<size_t>(actual_threads));
    for (int t = 0; t < actual_threads; ++t) workers.emplace_back(worker);
    for (auto& t : workers) t.join();
    if (failed.load()) throw std::runtime_error("Compact6 streaming shard worker failed: " + first_error);
    const auto shard_end = std::chrono::steady_clock::now();
    double shard_elapsed = std::chrono::duration<double>(shard_end - shard_start).count();

    const auto aggregate_start = std::chrono::steady_clock::now();
    uint64_t partial_shards = 0;
    uint64_t db_records = 0;
    uint64_t db_bytes = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        fs::path partial_path = partial_dir / (shard_name(shard) + ".partial.bin");
        fs::path done_json = partial_dir / (shard_name(shard) + ".done.json");
        fs::path compact_path = compact_dir / "shards" / (shard_name(shard) + ".compact6.bin");
        if (!fs::exists(partial_path) || !fs::exists(done_json)) {
            throw std::runtime_error("Missing completed compact6 partial shard: " + shard_name(shard));
        }
        add_partial_counts(partial_path, scores);
        ++partial_shards;
        db_bytes += fs::file_size(compact_path);
        db_records += fs::file_size(compact_path) / 6;
    }
    for (auto& score : scores) {
        uint16_t called = score.human_unique_kmers + score.bacteria_unique_kmers + score.fungi_unique_kmers + score.virus_unique_kmers;
        score.no_evidence_kmers = score.total_valid_kmers >= called ? score.total_valid_kmers - called : 0;
        finalize_score(score);
    }
    const auto aggregate_end = std::chrono::steady_clock::now();
    double aggregate_elapsed = std::chrono::duration<double>(aggregate_end - aggregate_start).count();

    const auto write_start = std::chrono::steady_clock::now();
    fs::create_directories(output_tsv.parent_path());
    fs::path tmp_path = output_tsv;
    tmp_path += ".tmp";
    std::ofstream out(tmp_path);
    if (!out) throw std::runtime_error("Could not write compact6 query TSV: " + tmp_path.string());
    out << "read_id\trow_index\terror_id\ttruth_domain\ttotal_valid_kmers\tno_evidence_kmers\t"
        << "human_unique_kmers\tbacteria_unique_kmers\tfungi_unique_kmers\tvirus_unique_kmers\tpred_domain\tpred_status\n";
    for (size_t i = 0; i < reads.size(); ++i) {
        const auto& r = reads[i];
        const auto& s = scores[i];
        out << r.read_id << '\t' << r.row_index << '\t' << r.error_id << '\t' << r.truth_domain << '\t'
            << s.total_valid_kmers << '\t' << s.no_evidence_kmers << '\t'
            << s.human_unique_kmers << '\t' << s.bacteria_unique_kmers << '\t'
            << s.fungi_unique_kmers << '\t' << s.virus_unique_kmers << '\t'
            << s.pred_domain << '\t' << s.pred_status << '\n';
    }
    out.close();
    fs::rename(tmp_path, output_tsv);
    const auto write_end = std::chrono::steady_clock::now();
    double write_elapsed = std::chrono::duration<double>(write_end - write_start).count();

    const auto end = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(end - start).count();
    double query_elapsed = event_elapsed + shard_elapsed + aggregate_elapsed;
    write_json_summary(done_path, {
        {"created_at", json_quote(now_string())},
        {"status", json_quote("completed")},
        {"mode", json_quote("query-compact6-domain-db-stream")},
        {"method_label", json_quote(method_label)},
        {"reads_tsv", json_quote(reads_tsv.string())},
        {"compact_dir", json_quote(compact_dir.string())},
        {"partial_dir", json_quote(partial_dir.string())},
        {"output_tsv", json_quote(output_tsv.string())},
        {"compact_shards", std::to_string(compact_shards)},
        {"threads", std::to_string(actual_threads)},
        {"reads", std::to_string(reads.size())},
        {"raw_kmers", std::to_string(raw_kmers)},
        {"partial_shards", std::to_string(partial_shards)},
        {"partial_shards_completed_before_start", std::to_string(completed_before_start)},
        {"db_records", std::to_string(db_records)},
        {"db_bytes", std::to_string(db_bytes)},
        {"event_generation_elapsed_seconds", std::to_string(event_elapsed)},
        {"shard_processing_elapsed_seconds", std::to_string(shard_elapsed)},
        {"aggregation_elapsed_seconds", std::to_string(aggregate_elapsed)},
        {"output_write_elapsed_seconds", std::to_string(write_elapsed)},
        {"query_elapsed_seconds", std::to_string(query_elapsed)},
        {"total_elapsed_seconds", std::to_string(total_elapsed)},
        {"reads_per_second_query_only", std::to_string(reads.empty() ? 0.0 : static_cast<double>(reads.size()) / query_elapsed)},
        {"kmer_lookups_per_second_query_only", std::to_string(raw_kmers == 0 ? 0.0 : static_cast<double>(raw_kmers) / query_elapsed)}
    });
    log_msg("Completed compact6 streaming query method=" + method_label + " reads=" + std::to_string(reads.size()) + " query_elapsed=" + std::to_string(query_elapsed));
}

void mode_query_reads(const Args& args) {
    const auto start = std::chrono::steady_clock::now();
    fs::path reads_tsv(require_arg(args, "reads-tsv"));
    fs::path compact_dir(require_arg(args, "compact-dir"));
    fs::path output_tsv(require_arg(args, "output-tsv"));
    int compact_shards = std::stoi(require_arg(args, "compact-shards"));
    int threads = std::stoi(require_arg(args, "threads"));
    int k = has_arg(args, "k") ? std::stoi(require_arg(args, "k")) : 25;
    uint64_t max_reads = has_arg(args, "max-reads") ? std::stoull(require_arg(args, "max-reads")) : 0;
    fs::path done_path = output_tsv;
    done_path += ".done.json";
    if (fs::exists(done_path) && fs::exists(output_tsv) && !has_arg(args, "force")) {
        log_msg("Compact query already completed, skipping: " + done_path.string());
        return;
    }

    std::vector<ReadRow> reads = load_reads(reads_tsv, max_reads);
    log_msg("Loaded reads=" + std::to_string(reads.size()));
    const auto load_start = std::chrono::steady_clock::now();
    std::vector<MMapShard> db(static_cast<size_t>(compact_shards));
    uint64_t db_records = 0;
    uint64_t db_bytes = 0;
    for (int shard = 0; shard < compact_shards; ++shard) {
        fs::path path = compact_dir / "shards" / (shard_name(shard) + ".compact.bin");
        db[static_cast<size_t>(shard)].open_file(path);
        db_records += db[static_cast<size_t>(shard)].records();
        db_bytes += db[static_cast<size_t>(shard)].size_bytes();
    }
    const auto load_end = std::chrono::steady_clock::now();
    double load_elapsed = std::chrono::duration<double>(load_end - load_start).count();
    log_msg("Mapped compact DB records=" + std::to_string(db_records) + " bytes=" + std::to_string(db_bytes));

    std::vector<ScoreRow> scores(reads.size());
    std::atomic<size_t> next_index{0};
    const auto query_start = std::chrono::steady_clock::now();
    auto worker = [&]() {
        KmerEncoder encoder(k);
        while (true) {
            size_t i = next_index.fetch_add(1);
            if (i >= reads.size()) break;
            ScoreRow local;
            local.total_valid_kmers = encoder.feed_sequence(reads[i].sequence, [&](uint64_t kmer) {
                int shard = static_cast<int>(kmer % static_cast<uint64_t>(compact_shards));
                uint8_t domain_id = db[static_cast<size_t>(shard)].lookup(kmer);
                if (domain_id == 1) ++local.human_unique_kmers;
                else if (domain_id == 2) ++local.bacteria_unique_kmers;
                else if (domain_id == 3) ++local.fungi_unique_kmers;
                else if (domain_id == 4) ++local.virus_unique_kmers;
                else ++local.no_evidence_kmers;
            });
            finalize_score(local);
            scores[i] = std::move(local);
        }
    };
    std::vector<std::thread> workers;
    int actual_threads = std::max(1, threads);
    workers.reserve(static_cast<size_t>(actual_threads));
    for (int t = 0; t < actual_threads; ++t) workers.emplace_back(worker);
    for (auto& t : workers) t.join();
    const auto query_end = std::chrono::steady_clock::now();
    double query_elapsed = std::chrono::duration<double>(query_end - query_start).count();

    fs::create_directories(output_tsv.parent_path());
    fs::path tmp_path = output_tsv;
    tmp_path += ".tmp";
    std::ofstream out(tmp_path);
    if (!out) throw std::runtime_error("Could not write compact query TSV: " + tmp_path.string());
    out << "read_id\trow_index\terror_id\ttruth_domain\ttotal_valid_kmers\tno_evidence_kmers\t"
        << "human_unique_kmers\tbacteria_unique_kmers\tfungi_unique_kmers\tvirus_unique_kmers\tpred_domain\tpred_status\n";
    uint64_t raw_kmers = 0;
    for (size_t i = 0; i < reads.size(); ++i) {
        const auto& r = reads[i];
        const auto& s = scores[i];
        raw_kmers += s.total_valid_kmers;
        out << r.read_id << '\t' << r.row_index << '\t' << r.error_id << '\t' << r.truth_domain << '\t'
            << s.total_valid_kmers << '\t' << s.no_evidence_kmers << '\t'
            << s.human_unique_kmers << '\t' << s.bacteria_unique_kmers << '\t'
            << s.fungi_unique_kmers << '\t' << s.virus_unique_kmers << '\t'
            << s.pred_domain << '\t' << s.pred_status << '\n';
    }
    out.close();
    fs::rename(tmp_path, output_tsv);

    const auto end = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(end - start).count();
    write_json_summary(done_path, {
        {"created_at", json_quote(now_string())},
        {"status", json_quote("completed")},
        {"mode", json_quote("query-compact-domain-db")},
        {"reads_tsv", json_quote(reads_tsv.string())},
        {"compact_dir", json_quote(compact_dir.string())},
        {"output_tsv", json_quote(output_tsv.string())},
        {"compact_shards", std::to_string(compact_shards)},
        {"threads", std::to_string(actual_threads)},
        {"reads", std::to_string(reads.size())},
        {"raw_kmers", std::to_string(raw_kmers)},
        {"db_records", std::to_string(db_records)},
        {"db_bytes", std::to_string(db_bytes)},
        {"load_elapsed_seconds", std::to_string(load_elapsed)},
        {"query_elapsed_seconds", std::to_string(query_elapsed)},
        {"total_elapsed_seconds", std::to_string(total_elapsed)},
        {"reads_per_second_query_only", std::to_string(reads.empty() ? 0.0 : static_cast<double>(reads.size()) / query_elapsed)},
        {"kmer_lookups_per_second_query_only", std::to_string(raw_kmers == 0 ? 0.0 : static_cast<double>(raw_kmers) / query_elapsed)}
    });
    log_msg("Completed compact query reads=" + std::to_string(reads.size()) + " query_elapsed=" + std::to_string(query_elapsed));
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        std::string mode = require_arg(args, "mode");
        if (mode == "build-shard") {
            mode_build_shard(args);
        } else if (mode == "convert-compact6-shard") {
            mode_convert_compact6_shard(args);
        } else if (mode == "query-reads-stream") {
            mode_query_reads_stream(args);
        } else if (mode == "query-reads-stream6") {
            mode_query_reads_stream6(args);
        } else if (mode == "query-reads") {
            mode_query_reads(args);
        } else {
            throw std::runtime_error("Unknown mode: " + mode);
        }
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << std::endl;
        return 1;
    }
    return 0;
}
