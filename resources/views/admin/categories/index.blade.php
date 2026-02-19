@extends('layouts.adminSB')
@section('title', 'Quickbite - Categories')

@section('content')
    <div class="container">
        <h2>Daftar Kategori</h2>
        <a href="{{ route('admin.categories.create') }}" class="btn btn-success mb-3">Tambah Kategori</a>

        @if (session('success'))
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                {{ session('success') }}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Tutup"></button>
            </div>
        @endif

        <table class="table table-bordered">
            <thead class="table-light">
                <tr>
                    <th>ID</th>
                    <th>Nama</th>
                    <th>Gambar</th>
                    <th>Dibuat</th>
                    <th>Diubah</th>
                    <th>Aksi</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($listCategory as $category)
                    <tr>
                        <td class="text-center">{{ $category->id }}</td>
                        <td>{{ $category->name }}</td>
                        <td>
                            <img src="{{ asset('storage/' . $category->image_path) }}" alt="{{ $category->name }}"
                                style="width: 120px; height: 120px; object-fit: cover; border-radius: 6px;" />
                        </td>
                        <td>{{ $category->created_at }}</td>
                        <td>{{ $category->updated_at }}</td>
                        <td class="text-center">
                            <a href="#modalEdit" class="btn btn-primary btn-sm" data-bs-toggle="modal"
                                onclick="getEditForm({{ $category->id }})">
                                Edit
                            </a>
                            <button class="btn btn-danger btn-sm" data-bs-toggle="modal"
                                data-bs-target="#deleteModal{{ $category->id }}">
                                Delete
                            </button>
                        </td>
                    </tr>
                    @push('modals')
                        <!-- Delete Modal -->
                        <div class="modal fade" id="deleteModal{{ $category->id }}" tabindex="-1"
                            aria-labelledby="deleteModalLabel{{ $category->id }}" aria-hidden="true">
                            <div class="modal-dialog">
                                <div class="modal-content">
                                    <div class="modal-header bg-danger text-white">
                                        <h5 class="modal-title" id="deleteModalLabel{{ $category->id }}">Konfirmasi Hapus</h5>
                                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Tutup"></button>
                                    </div>
                                    <div class="modal-body">
                                        Apakah kamu yakin ingin menghapus kategori <strong>{{ $category->name }}</strong>?
                                    </div>
                                    <div class="modal-footer">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Batal</button>
                                        <form action="{{ route('admin.categories.destroy', $category->id) }}" method="POST">
                                            @csrf
                                            @method('DELETE')
                                            <button type="submit" class="btn btn-danger">Hapus</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                        </div>
                    @endpush
                @endforeach
            </tbody>
        </table>
    </div>
    @push('modals')
        <div class="modal fade" id="modalEdit" tabindex="-1" role="basic" aria-hidden="true">
            <div class="modal-dialog modal-wide">
                <div class="modal-content">
                    <div class="modal-body" id="modalContent">
                        {{-- You can put animated loading image here... --}}
                        <div class="d-flex justify-content-center align-items-center" style="height: 200px;">
                            <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    @endpush
    @push('scripts')
        <script>
            function getEditForm(id) {
                $.ajax({
                    type: 'POST',
                    url: '{{route("admin.categories.getEditForm")}}',
                    data: {
                        '_token': '<?php echo csrf_token() ?>',
                        'id': id
                    },
                    success: function (data) {
                        $('#modalContent').html(data.msg)
                    }
                });
            }
        </script>
    @endpush
@endsection